use std::{
    error::Error,
    fmt::{Display, Formatter},
    fs,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver},
    thread,
    time::Instant,
};

use ann::{
    module::{AutodiffModule, Module},
    optim::{AdamConfig, GradientsParams, Optimizer},
    prelude::Backend,
    record::DefaultRecorder,
    tensor::{Distribution, ElementConversion, Tensor, backend::AutodiffBackend},
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

use super::{
    data::{CocoPoseDataset, PoseDataError, PoseTensorBatch},
    loss::joints_mse_loss,
    model::{LiteHrNetPose, LiteHrNetPoseConfig},
};

pub struct PoseBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 4>,
    pub target_weight: Tensor<B, 3>,
}

pub fn synthetic_pose_batch<B: Backend>(
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
    let (model, loss) = train_step_with_loss_tensor(model, optimizer, batch, learning_rate);
    (model, loss_to_f64(loss))
}

pub fn train_step_with_loss_tensor<B, O>(
    model: LiteHrNetPose<B>,
    optimizer: &mut O,
    batch: PoseBatch<B>,
    learning_rate: f64,
) -> (LiteHrNetPose<B>, Tensor<B, 1>)
where
    B: AutodiffBackend,
    O: Optimizer<LiteHrNetPose<B>, B>,
{
    let predictions = model.forward(batch.images);
    let loss = joints_mse_loss(predictions, batch.targets, Some(batch.target_weight));
    let loss_value = loss.clone().detach();
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
    pub prefetch_batches: usize,
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
            prefetch_batches: 0,
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

        let mut train_loss_sum = None;
        let mut train_batches = 0;
        let mut train_samples = 0;

        let mut batch_loader = TensorBatchLoader::new(
            &train_dataset,
            &indices,
            config.batch_size,
            config.prefetch_batches,
        );

        while let Some(tensor_batch) = batch_loader.next_batch()? {
            let batch_samples = tensor_batch.len();
            let batch = tensor_batch.into_pose_batch::<B>(device);
            let (updated_model, loss) =
                train_step_with_loss_tensor(model, &mut optimizer, batch, config.learning_rate);
            model = updated_model;
            train_loss_sum = Some(accumulate_weighted_loss(
                train_loss_sum,
                loss,
                batch_samples,
            ));
            train_batches += 1;
            train_samples += batch_samples;

            if config.log_every > 0 && train_batches % config.log_every == 0 {
                let train_loss = average_loss(
                    train_loss_sum
                        .as_ref()
                        .expect("training loss exists after first batch"),
                    train_samples,
                );
                progress(PoseTrainingProgress::Batch(BatchTrainingProgress {
                    epoch,
                    train_batches,
                    train_samples,
                    train_loss,
                }));
            }
        }

        let train_loss = average_loss(
            train_loss_sum
                .as_ref()
                .expect("training loss exists after first batch"),
            train_samples,
        );
        let val_loss = match &val_dataset {
            Some(dataset) => Some(evaluate_dataset_inner(
                &model.valid(),
                dataset,
                config.batch_size,
                config.prefetch_batches,
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
    evaluate_dataset_inner(&model.valid(), dataset, batch_size, 0, device)
}

fn evaluate_dataset_inner<B: Backend>(
    model: &LiteHrNetPose<B>,
    dataset: &CocoPoseDataset,
    batch_size: usize,
    prefetch_batches: usize,
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
    let mut loss_sum = None;
    let mut samples = 0;

    let mut batch_loader = TensorBatchLoader::new(dataset, &indices, batch_size, prefetch_batches);
    while let Some(tensor_batch) = batch_loader.next_batch()? {
        let batch_samples = tensor_batch.len();
        let batch = tensor_batch.into_pose_batch::<B>(device);
        let predictions = model.forward(batch.images);
        let loss = joints_mse_loss(predictions, batch.targets, Some(batch.target_weight));
        loss_sum = Some(accumulate_weighted_loss(
            loss_sum,
            loss.detach(),
            batch_samples,
        ));
        samples += batch_samples;
    }

    Ok(average_loss(
        loss_sum
            .as_ref()
            .expect("validation loss exists after first batch"),
        samples,
    ))
}

fn accumulate_weighted_loss<B: Backend>(
    loss_sum: Option<Tensor<B, 1>>,
    loss: Tensor<B, 1>,
    samples: usize,
) -> Tensor<B, 1> {
    let weighted = loss * samples as f64;
    match loss_sum {
        Some(loss_sum) => loss_sum + weighted,
        None => weighted,
    }
}

fn average_loss<B: Backend>(loss_sum: &Tensor<B, 1>, samples: usize) -> f64 {
    loss_to_f64(loss_sum.clone() / samples as f64)
}

fn loss_to_f64<B: Backend>(loss: Tensor<B, 1>) -> f64 {
    loss.into_scalar().elem::<f64>()
}

enum TensorBatchLoader<'a> {
    Direct {
        dataset: &'a CocoPoseDataset,
        chunks: std::slice::Chunks<'a, usize>,
    },
    Prefetch(PrefetchedTensorBatches),
}

impl<'a> TensorBatchLoader<'a> {
    fn new(
        dataset: &'a CocoPoseDataset,
        indices: &'a [usize],
        batch_size: usize,
        prefetch_batches: usize,
    ) -> Self {
        if prefetch_batches == 0 {
            Self::Direct {
                dataset,
                chunks: indices.chunks(batch_size),
            }
        } else {
            Self::Prefetch(PrefetchedTensorBatches::new(
                dataset.clone(),
                indices.to_vec(),
                batch_size,
                prefetch_batches,
            ))
        }
    }

    fn next_batch(&mut self) -> Result<Option<PoseTensorBatch>, PoseTrainError> {
        match self {
            Self::Direct { dataset, chunks } => match chunks.next() {
                Some(indices) => Ok(Some(dataset.load_tensor_batch(indices)?)),
                None => Ok(None),
            },
            Self::Prefetch(prefetch) => prefetch.next_batch(),
        }
    }
}

struct PrefetchedTensorBatches {
    receiver: Receiver<Result<PoseTensorBatch, PoseDataError>>,
    remaining: usize,
}

impl PrefetchedTensorBatches {
    fn new(
        dataset: CocoPoseDataset,
        indices: Vec<usize>,
        batch_size: usize,
        prefetch_batches: usize,
    ) -> Self {
        let total_batches = indices.len().div_ceil(batch_size);
        let (sender, receiver) = mpsc::sync_channel(prefetch_batches);

        thread::spawn(move || {
            for batch_indices in indices.chunks(batch_size) {
                let result = dataset.load_tensor_batch(batch_indices);
                let is_error = result.is_err();
                if sender.send(result).is_err() || is_error {
                    break;
                }
            }
        });

        Self {
            receiver,
            remaining: total_batches,
        }
    }

    fn next_batch(&mut self) -> Result<Option<PoseTensorBatch>, PoseTrainError> {
        if self.remaining == 0 {
            return Ok(None);
        }

        self.remaining -= 1;
        let batch = self
            .receiver
            .recv()
            .map_err(|_| PoseTrainError::Prefetch("batch prefetch worker stopped".to_string()))??;
        Ok(Some(batch))
    }
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
    Recorder(ann::record::RecorderError),
    InvalidConfig(String),
    Prefetch(String),
}

impl Display for PoseTrainError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(error) => write!(formatter, "data error: {error}"),
            Self::Io(error) => write!(formatter, "io error: {error}"),
            Self::Json(error) => write!(formatter, "json error: {error}"),
            Self::Recorder(error) => write!(formatter, "recorder error: {error}"),
            Self::InvalidConfig(message) => write!(formatter, "invalid config: {message}"),
            Self::Prefetch(message) => write!(formatter, "prefetch error: {message}"),
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

extern crate ann as burn;
