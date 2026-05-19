use std::{
    error::Error,
    fmt::{Display, Formatter},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use burn::{
    module::Module,
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::DefaultRecorder,
    tensor::{Distribution, ElementConversion, Tensor, backend::AutodiffBackend},
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

use crate::{CocoPoseDataset, LiteHrNetPose, LiteHrNetPoseConfig, PoseDataError, joints_mse_loss};

pub struct PoseBatch<B: AutodiffBackend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
    pub target_weight: Tensor<B, 3>,
}

pub fn synthetic_pose_batch<B: AutodiffBackend>(
    batch_size: usize,
    image_height: usize,
    image_width: usize,
    num_joints: usize,
    device: &B::Device,
) -> PoseBatch<B> {
    PoseBatch {
        images: Tensor::random(
            [batch_size, 3, image_height, image_width],
            Distribution::Normal(0.0, 1.0),
            device,
        ),
        targets: Tensor::zeros(
            [batch_size, num_joints, image_height / 4, image_width / 4],
            device,
        ),
        target_weight: Tensor::ones([batch_size, num_joints, 1], device),
    }
}

pub fn train_step<B, O>(
    model: LiteHrNetPose<B>,
    optimizer: &mut O,
    batch: PoseBatch<B>,
    learning_rate: f64,
) -> LiteHrNetPose<B>
where
    B: AutodiffBackend,
    O: Optimizer<LiteHrNetPose<B>, B>,
{
    train_step_with_loss(model, optimizer, batch, learning_rate).0
}

pub fn train_step_with_loss<B, O>(
    model: LiteHrNetPose<B>,
    optimizer: &mut O,
    batch: PoseBatch<B>,
    learning_rate: f64,
) -> (LiteHrNetPose<B>, f64)
where
    B: AutodiffBackend,
    O: Optimizer<LiteHrNetPose<B>, B>,
{
    let predictions = model.forward(batch.images);
    let loss = joints_mse_loss(predictions, batch.targets, Some(batch.target_weight));
    let loss_value = loss.clone().into_scalar().elem::<f64>();
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    (optimizer.step(learning_rate, model, grads), loss_value)
}

pub fn run_synthetic_training<B: AutodiffBackend>(
    config: LiteHrNetPoseConfig,
    device: &B::Device,
    steps: usize,
    batch_size: usize,
    image_height: usize,
    image_width: usize,
    learning_rate: f64,
) -> LiteHrNetPose<B> {
    let mut model = config.init(device);
    let mut optimizer = AdamConfig::new().init::<B, LiteHrNetPose<B>>();

    for _ in 0..steps {
        let batch = synthetic_pose_batch(
            batch_size,
            image_height,
            image_width,
            config.num_joints,
            device,
        );
        model = train_step(model, &mut optimizer, batch, learning_rate);
    }

    model
}

#[derive(Clone, Debug)]
pub struct PoseTrainingConfig {
    pub model: LiteHrNetPoseConfig,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub shuffle: bool,
    pub seed: u64,
    pub max_train_samples: Option<usize>,
    pub max_val_samples: Option<usize>,
    pub checkpoint_dir: PathBuf,
    pub log_every: usize,
    pub save_every_epoch: bool,
}

impl PoseTrainingConfig {
    pub fn coco_default(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            model: LiteHrNetPoseConfig::litehrnet18_coco(),
            epochs: 210,
            batch_size: 32,
            learning_rate: 2e-3,
            shuffle: true,
            seed: 42,
            max_train_samples: None,
            max_val_samples: None,
            checkpoint_dir: checkpoint_dir.into(),
            log_every: 50,
            save_every_epoch: true,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct EpochReport {
    pub epoch: usize,
    pub train_loss: f64,
    pub val_loss: Option<f64>,
    pub train_batches: usize,
    pub train_samples: usize,
    pub elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PoseTrainingReport {
    pub epochs: Vec<EpochReport>,
    pub best_val_loss: Option<f64>,
}

impl PoseTrainingReport {
    fn new() -> Self {
        Self {
            epochs: Vec::new(),
            best_val_loss: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BatchTrainingProgress {
    pub epoch: usize,
    pub train_batches: usize,
    pub train_samples: usize,
    pub train_loss: f64,
}

#[derive(Clone, Debug)]
pub enum PoseTrainingProgress {
    Batch(BatchTrainingProgress),
    Epoch(EpochReport),
}

pub fn train_dataset<B: AutodiffBackend>(
    config: PoseTrainingConfig,
    train_dataset: CocoPoseDataset,
    val_dataset: Option<CocoPoseDataset>,
    device: &B::Device,
) -> Result<(LiteHrNetPose<B>, PoseTrainingReport), PoseTrainError> {
    train_dataset_with_progress(config, train_dataset, val_dataset, device, |_| {})
}

pub fn train_dataset_with_progress<B, F>(
    config: PoseTrainingConfig,
    train_dataset: CocoPoseDataset,
    val_dataset: Option<CocoPoseDataset>,
    device: &B::Device,
    mut progress: F,
) -> Result<(LiteHrNetPose<B>, PoseTrainingReport), PoseTrainError>
where
    B: AutodiffBackend,
    F: FnMut(PoseTrainingProgress),
{
    if config.batch_size == 0 {
        return Err(PoseTrainError::InvalidConfig(
            "batch_size must be greater than zero".to_string(),
        ));
    }
    if config.epochs == 0 {
        return Err(PoseTrainError::InvalidConfig(
            "epochs must be greater than zero".to_string(),
        ));
    }

    fs::create_dir_all(&config.checkpoint_dir)?;
    let train_dataset = train_dataset.limited(config.max_train_samples);
    let val_dataset = val_dataset.map(|dataset| dataset.limited(config.max_val_samples));
    if train_dataset.is_empty() {
        return Err(PoseTrainError::InvalidConfig(
            "training dataset is empty after applying max_train_samples".to_string(),
        ));
    }
    if val_dataset.as_ref().is_some_and(CocoPoseDataset::is_empty) {
        return Err(PoseTrainError::InvalidConfig(
            "validation dataset is empty after applying max_val_samples".to_string(),
        ));
    }
    let mut indices = (0..train_dataset.len()).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut model = config.model.init(device);
    let mut optimizer = AdamConfig::new().init::<B, LiteHrNetPose<B>>();
    let mut report = PoseTrainingReport::new();

    for epoch in 1..=config.epochs {
        let started = Instant::now();
        if config.shuffle {
            indices.shuffle(&mut rng);
        }

        let mut train_loss_sum = 0.0;
        let mut train_batches = 0;
        let mut train_samples = 0;

        for batch_indices in indices.chunks(config.batch_size) {
            let batch = train_dataset.batch::<B>(batch_indices, device)?;
            let (updated_model, loss) =
                train_step_with_loss(model, &mut optimizer, batch, config.learning_rate);
            model = updated_model;
            train_loss_sum += loss * batch_indices.len() as f64;
            train_batches += 1;
            train_samples += batch_indices.len();

            if config.log_every > 0 && train_batches % config.log_every == 0 {
                progress(PoseTrainingProgress::Batch(BatchTrainingProgress {
                    epoch,
                    train_batches,
                    train_samples,
                    train_loss: train_loss_sum / train_samples as f64,
                }));
            }
        }

        let train_loss = train_loss_sum / train_samples as f64;
        let val_loss = match &val_dataset {
            Some(dataset) => Some(evaluate_dataset(
                &model,
                dataset,
                config.batch_size,
                device,
            )?),
            None => None,
        };

        let epoch_report = EpochReport {
            epoch,
            train_loss,
            val_loss,
            train_batches,
            train_samples,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        };

        progress(PoseTrainingProgress::Epoch(epoch_report.clone()));

        if config.save_every_epoch {
            save_model_checkpoint(&model, &config.checkpoint_dir, &format!("epoch_{epoch:03}"))?;
        }
        save_model_checkpoint(&model, &config.checkpoint_dir, "last")?;

        if let Some(val_loss) = epoch_report.val_loss {
            let is_best = report.best_val_loss.is_none_or(|best| val_loss < best);
            if is_best {
                report.best_val_loss = Some(val_loss);
                save_model_checkpoint(&model, &config.checkpoint_dir, "best")?;
            }
        }

        report.epochs.push(epoch_report);
        write_report(&report, config.checkpoint_dir.join("training_report.json"))?;
    }

    Ok((model, report))
}

pub fn evaluate_dataset<B: AutodiffBackend>(
    model: &LiteHrNetPose<B>,
    dataset: &CocoPoseDataset,
    batch_size: usize,
    device: &B::Device,
) -> Result<f64, PoseTrainError> {
    if batch_size == 0 {
        return Err(PoseTrainError::InvalidConfig(
            "batch_size must be greater than zero".to_string(),
        ));
    }
    if dataset.is_empty() {
        return Err(PoseTrainError::InvalidConfig(
            "validation dataset is empty".to_string(),
        ));
    }

    let indices = (0..dataset.len()).collect::<Vec<_>>();
    let mut loss_sum = 0.0;
    let mut samples = 0;

    for batch_indices in indices.chunks(batch_size) {
        let batch = dataset.batch::<B>(batch_indices, device)?;
        let predictions = model.forward(batch.images);
        let loss = joints_mse_loss(predictions, batch.targets, Some(batch.target_weight));
        loss_sum += loss.into_scalar().elem::<f64>() * batch_indices.len() as f64;
        samples += batch_indices.len();
    }

    Ok(loss_sum / samples as f64)
}

fn save_model_checkpoint<B: AutodiffBackend>(
    model: &LiteHrNetPose<B>,
    checkpoint_dir: &Path,
    name: &str,
) -> Result<(), PoseTrainError> {
    let recorder = DefaultRecorder::default();
    model
        .clone()
        .save_file(checkpoint_dir.join(name), &recorder)
        .map_err(PoseTrainError::Recorder)
}

fn write_report(report: &PoseTrainingReport, path: impl AsRef<Path>) -> Result<(), PoseTrainError> {
    let json = json::to_string_pretty(report)?;
    fs::write(path, json)?;
    Ok(())
}

#[derive(Debug)]
pub enum PoseTrainError {
    Data(PoseDataError),
    Io(std::io::Error),
    Json(json::Error),
    Recorder(burn::record::RecorderError),
    InvalidConfig(String),
}

impl Display for PoseTrainError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(error) => write!(formatter, "data error: {error}"),
            Self::Io(error) => write!(formatter, "io error: {error}"),
            Self::Json(error) => write!(formatter, "json error: {error}"),
            Self::Recorder(error) => write!(formatter, "recorder error: {error}"),
            Self::InvalidConfig(message) => write!(formatter, "invalid config: {message}"),
        }
    }
}

impl Error for PoseTrainError {}

impl From<PoseDataError> for PoseTrainError {
    fn from(value: PoseDataError) -> Self {
        Self::Data(value)
    }
}

impl From<std::io::Error> for PoseTrainError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<json::Error> for PoseTrainError {
    fn from(value: json::Error) -> Self {
        Self::Json(value)
    }
}
