use std::{fs, path::PathBuf, time::Instant};

use ann::{
    optim::{AdamConfig, GradientsParams, Optimizer},
    tensor::{
        ElementConversion, Tensor, TensorData, activation, backend::AutodiffBackend,
        backend::Backend,
    },
};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::Serialize;

use super::{
    RetrievalError, RetrievalModel, RetrievalModelConfig, RetrievalPairDataset, ensure_feature_dim,
    extract_glyph_features_with_cache, extract_pose_features_with_cache, save_retrieval_model,
    write_json_file,
};

#[derive(Clone, Debug)]
pub struct RetrievalTrainingConfig {
    pub model: RetrievalModelConfig,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    pub temperature: f64,
    pub shuffle: bool,
    pub seed: u64,
    pub max_pairs: Option<usize>,
    pub checkpoint_dir: PathBuf,
    pub log_every: usize,
    pub save_every_epoch: bool,
}

impl RetrievalTrainingConfig {
    pub fn default_with_checkpoint_dir(checkpoint_dir: impl Into<PathBuf>) -> Self {
        Self {
            model: RetrievalModelConfig::default(),
            epochs: 20,
            batch_size: 32,
            learning_rate: 1e-3,
            temperature: 0.07,
            shuffle: true,
            seed: 42,
            max_pairs: None,
            checkpoint_dir: checkpoint_dir.into(),
            log_every: 20,
            save_every_epoch: false,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct RetrievalEpochReport {
    pub epoch: usize,
    pub train_loss: f64,
    pub train_batches: usize,
    pub train_pairs: usize,
    pub elapsed_seconds: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RetrievalTrainingReport {
    pub epochs: Vec<RetrievalEpochReport>,
}

#[derive(Clone, Debug)]
pub struct RetrievalBatchProgress {
    pub epoch: usize,
    pub train_batches: usize,
    pub train_pairs: usize,
    pub train_loss: f64,
}

#[derive(Clone, Debug)]
pub enum RetrievalTrainingProgress {
    Batch(RetrievalBatchProgress),
    Epoch(RetrievalEpochReport),
}

pub fn train_retrieval_dataset<B, F>(
    config: RetrievalTrainingConfig,
    dataset: &RetrievalPairDataset,
    device: &B::Device,
    mut progress: F,
) -> Result<(RetrievalModel<B>, RetrievalTrainingReport), RetrievalError>
where
    B: AutodiffBackend,
    F: FnMut(RetrievalTrainingProgress),
{
    if config.epochs == 0 {
        return Err(RetrievalError::InvalidData(
            "epochs must be greater than zero".to_string(),
        ));
    }
    if config.batch_size == 0 {
        return Err(RetrievalError::InvalidData(
            "batch_size must be greater than zero".to_string(),
        ));
    }
    if config.temperature <= 0.0 || !config.temperature.is_finite() {
        return Err(RetrievalError::InvalidData(
            "temperature must be a finite positive number".to_string(),
        ));
    }

    fs::create_dir_all(&config.checkpoint_dir)?;
    write_json_file(
        config.checkpoint_dir.join("retrieval_config.json"),
        &config.model,
    )?;

    let feature_cache_dir = config.checkpoint_dir.join("feature_cache");
    let pairs = load_feature_pairs(dataset, config.max_pairs, Some(&feature_cache_dir))?;
    if pairs.len() < 2 {
        return Err(RetrievalError::InvalidData(
            "contrastive retrieval training requires at least two pairs".to_string(),
        ));
    }

    let mut indices = (0..pairs.len()).collect::<Vec<_>>();
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut model = config.model.init(device);
    let mut optimizer = AdamConfig::new().init::<B, RetrievalModel<B>>();
    let mut report = RetrievalTrainingReport { epochs: Vec::new() };

    for epoch in 1..=config.epochs {
        let started = Instant::now();
        if config.shuffle {
            indices.shuffle(&mut rng);
        }

        let mut loss_sum = 0.0;
        let mut train_batches = 0;
        let mut train_pairs = 0;

        for batch_indices in indices.chunks(config.batch_size) {
            let batch =
                retrieval_batch::<B>(&pairs, batch_indices, device, config.model.input_dim)?;
            let (updated, loss) = retrieval_train_step(
                model,
                &mut optimizer,
                batch,
                config.learning_rate,
                config.temperature,
            );
            model = updated;
            loss_sum += loss * batch_indices.len() as f64;
            train_batches += 1;
            train_pairs += batch_indices.len();

            if config.log_every > 0 && train_batches % config.log_every == 0 {
                progress(RetrievalTrainingProgress::Batch(RetrievalBatchProgress {
                    epoch,
                    train_batches,
                    train_pairs,
                    train_loss: loss_sum / train_pairs as f64,
                }));
            }
        }

        let epoch_report = RetrievalEpochReport {
            epoch,
            train_loss: loss_sum / train_pairs as f64,
            train_batches,
            train_pairs,
            elapsed_seconds: started.elapsed().as_secs_f64(),
        };
        progress(RetrievalTrainingProgress::Epoch(epoch_report.clone()));

        if config.save_every_epoch {
            save_retrieval_model(&model, &config.checkpoint_dir, &format!("epoch_{epoch:03}"))?;
        }
        save_retrieval_model(&model, &config.checkpoint_dir, "last")?;

        report.epochs.push(epoch_report);
        write_json_file(
            config.checkpoint_dir.join("retrieval_training_report.json"),
            &report,
        )?;
    }

    Ok((model, report))
}

#[derive(Clone, Debug)]
struct FeaturePair {
    pose: Vec<f32>,
    glyph: Vec<f32>,
}

struct RetrievalBatch<B: AutodiffBackend> {
    pose: Tensor<B, 2>,
    glyph: Tensor<B, 2>,
    labels: Tensor<B, 2>,
}

fn load_feature_pairs(
    dataset: &RetrievalPairDataset,
    max_pairs: Option<usize>,
    feature_cache_dir: Option<&std::path::Path>,
) -> Result<Vec<FeaturePair>, RetrievalError> {
    dataset
        .limited_pairs(max_pairs)
        .iter()
        .map(|pair| {
            Ok(FeaturePair {
                pose: extract_pose_features_with_cache(&pair.image_path, feature_cache_dir)?,
                glyph: extract_glyph_features_with_cache(&pair.glyph_path, feature_cache_dir)?,
            })
        })
        .collect()
}

fn retrieval_batch<B: AutodiffBackend>(
    pairs: &[FeaturePair],
    indices: &[usize],
    device: &B::Device,
    input_dim: usize,
) -> Result<RetrievalBatch<B>, RetrievalError> {
    if indices.is_empty() {
        return Err(RetrievalError::InvalidData(
            "retrieval batch requires at least one pair".to_string(),
        ));
    }

    let mut pose = Vec::with_capacity(indices.len() * input_dim);
    let mut glyph = Vec::with_capacity(indices.len() * input_dim);
    let mut labels = vec![0.0_f32; indices.len() * indices.len()];

    for (row, &index) in indices.iter().enumerate() {
        let pair = pairs.get(index).ok_or_else(|| {
            RetrievalError::InvalidData(format!("feature pair index {index} out of range"))
        })?;
        ensure_feature_dim(pair.pose.len(), input_dim)?;
        ensure_feature_dim(pair.glyph.len(), input_dim)?;
        pose.extend_from_slice(&pair.pose);
        glyph.extend_from_slice(&pair.glyph);
        labels[row * indices.len() + row] = 1.0;
    }

    Ok(RetrievalBatch {
        pose: Tensor::from_data(TensorData::new(pose, [indices.len(), input_dim]), device),
        glyph: Tensor::from_data(TensorData::new(glyph, [indices.len(), input_dim]), device),
        labels: Tensor::from_data(
            TensorData::new(labels, [indices.len(), indices.len()]),
            device,
        ),
    })
}

fn retrieval_train_step<B, O>(
    model: RetrievalModel<B>,
    optimizer: &mut O,
    batch: RetrievalBatch<B>,
    learning_rate: f64,
    temperature: f64,
) -> (RetrievalModel<B>, f64)
where
    B: AutodiffBackend,
    O: Optimizer<RetrievalModel<B>, B>,
{
    let pose_embedding = model.forward_pose(batch.pose);
    let glyph_embedding = model.forward_glyph(batch.glyph);
    let scores = pose_embedding
        .clone()
        .matmul(glyph_embedding.clone().transpose())
        .div_scalar(temperature);
    let reverse_scores = scores.clone().transpose();
    let labels = batch.labels;
    let loss_forward = contrastive_loss(scores, labels.clone());
    let loss_reverse = contrastive_loss(reverse_scores, labels.transpose());
    let loss = (loss_forward + loss_reverse).div_scalar(2.0);
    let loss_value = loss.clone().into_scalar().elem::<f64>();
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    (optimizer.step(learning_rate, model, grads), loss_value)
}

fn contrastive_loss<B: Backend>(scores: Tensor<B, 2>, labels: Tensor<B, 2>) -> Tensor<B, 1> {
    let batch = scores.dims()[0] as f64;
    let log_probs = activation::log_softmax(scores, 1);
    (log_probs * labels).sum().neg().div_scalar(batch)
}
