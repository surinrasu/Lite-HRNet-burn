use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::{Instant, UNIX_EPOCH},
};

use ann::{
    module::Module,
    nn::{Linear, LinearConfig},
    optim::{AdamConfig, GradientsParams, Optimizer},
    record::DefaultRecorder,
    tensor::{
        ElementConversion, Tensor, TensorData, activation, backend::AutodiffBackend,
        backend::Backend,
    },
};
use image::{DynamicImage, GenericImageView, ImageReader};
use rand::{SeedableRng, rngs::StdRng, seq::SliceRandom};
use serde::{Deserialize, Serialize, de::DeserializeOwned};

use crate::pose_estimation::{
    DefaultPoseEstimator, SPINEPOSE_FEATURE_DIM, SPINEPOSE_KEYPOINTS,
    estimate_pose_features_from_bytes, estimate_pose_features_from_path,
    find_spinepose_json_for_image,
};

mod error;
mod index;

pub use error::*;
pub use index::*;

pub const RETRIEVAL_KEYPOINTS: usize = SPINEPOSE_KEYPOINTS;
pub const RETRIEVAL_FEATURE_DIM: usize = SPINEPOSE_FEATURE_DIM;
pub const DEFAULT_RETRIEVAL_HIDDEN_DIM: usize = 128;
pub const DEFAULT_RETRIEVAL_EMBEDDING_DIM: usize = 64;
pub const CANDIDATE_INDEX_VERSION: u32 = 1;

const FEATURE_CACHE_VERSION: u32 = 1;
const FEATURE_POINTS_PER_BAND: usize = 3;
const FEATURE_BANDS: usize = RETRIEVAL_KEYPOINTS.div_ceil(FEATURE_POINTS_PER_BAND);

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalPair {
    pub id: String,
    pub codepoint: Option<String>,
    pub character: Option<String>,
    pub persona: String,
    pub image_path: PathBuf,
    pub glyph_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct RetrievalPairDataset {
    pairs: Vec<RetrievalPair>,
}

impl RetrievalPairDataset {
    pub fn from_data_root(data_root: impl AsRef<Path>) -> Result<Self, RetrievalError> {
        let data_root = resolve_existing_data_root(data_root.as_ref())?;
        let mut persona_dirs = Vec::new();

        for entry in fs::read_dir(&data_root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("persona_") {
                persona_dirs.push((name, entry.path()));
            }
        }
        persona_dirs.sort_by(|left, right| left.0.cmp(&right.0));

        let mut pairs = Vec::new();
        for (persona, persona_dir) in persona_dirs {
            let image_dir = persona_dir.join("images");
            let glyph_dir = persona_dir.join("glyphs");
            if !image_dir.is_dir() || !glyph_dir.is_dir() {
                continue;
            }

            let mut image_paths = Vec::new();
            for entry in fs::read_dir(&image_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() && is_png(&entry.path()) {
                    image_paths.push(entry.path());
                }
            }
            image_paths.sort();

            for image_path in image_paths {
                let Some(file_name) = image_path.file_name() else {
                    continue;
                };
                let glyph_path = glyph_dir.join(file_name);
                if !glyph_path.is_file() {
                    continue;
                }

                let Some(id) = image_path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(str::to_owned)
                else {
                    continue;
                };
                let metadata = parse_glyph_id(&id);
                pairs.push(RetrievalPair {
                    id,
                    codepoint: metadata.codepoint,
                    character: metadata.character,
                    persona: persona.clone(),
                    image_path: canonical_or_original(image_path),
                    glyph_path: canonical_or_original(glyph_path),
                });
            }
        }

        if pairs.is_empty() {
            return Err(RetrievalError::InvalidData(format!(
                "no image/glyph pairs found under {}",
                data_root.display()
            )));
        }

        Ok(Self { pairs })
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub fn pairs(&self) -> &[RetrievalPair] {
        &self.pairs
    }

    pub fn limited_pairs(&self, max_pairs: Option<usize>) -> Vec<RetrievalPair> {
        match max_pairs {
            Some(max_pairs) => self.pairs.iter().take(max_pairs).cloned().collect(),
            None => self.pairs.clone(),
        }
    }

    pub fn glyph_candidates(&self, unique_by_id: bool) -> Vec<GlyphCandidate> {
        if !unique_by_id {
            return self
                .pairs
                .iter()
                .map(GlyphCandidate::from_pair)
                .collect::<Vec<_>>();
        }

        let mut by_id = BTreeMap::new();
        for pair in &self.pairs {
            by_id
                .entry(pair.id.clone())
                .or_insert_with(|| GlyphCandidate::from_pair(pair));
        }
        by_id.into_values().collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlyphCandidate {
    pub id: String,
    pub codepoint: Option<String>,
    pub character: Option<String>,
    pub persona: String,
    pub glyph_path: PathBuf,
}

impl GlyphCandidate {
    fn from_pair(pair: &RetrievalPair) -> Self {
        Self {
            id: pair.id.clone(),
            codepoint: pair.codepoint.clone(),
            character: pair.character.clone(),
            persona: pair.persona.clone(),
            glyph_path: pair.glyph_path.clone(),
        }
    }
}

pub fn extract_pose_features_from_path(path: impl AsRef<Path>) -> Result<Vec<f32>, RetrievalError> {
    estimate_pose_features_from_path(path)
}

pub fn extract_glyph_features_from_path(
    path: impl AsRef<Path>,
) -> Result<Vec<f32>, RetrievalError> {
    let image = ImageReader::open(path)?.decode()?;
    Ok(extract_shape_features(&image))
}

pub fn extract_pose_features_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, RetrievalError> {
    estimate_pose_features_from_bytes(bytes)
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RetrievalModelConfig {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub embedding_dim: usize,
}

impl Default for RetrievalModelConfig {
    fn default() -> Self {
        Self {
            input_dim: RETRIEVAL_FEATURE_DIM,
            hidden_dim: DEFAULT_RETRIEVAL_HIDDEN_DIM,
            embedding_dim: DEFAULT_RETRIEVAL_EMBEDDING_DIM,
        }
    }
}

impl RetrievalModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> RetrievalModel<B> {
        RetrievalModel {
            pose_tower: RetrievalTower::new(
                self.input_dim,
                self.hidden_dim,
                self.embedding_dim,
                device,
            ),
            glyph_tower: RetrievalTower::new(
                self.input_dim,
                self.hidden_dim,
                self.embedding_dim,
                device,
            ),
            config: self.clone(),
        }
    }
}

#[derive(Module, Debug)]
pub struct RetrievalTower<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
}

impl<B: Backend> RetrievalTower<B> {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        embedding_dim: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            fc1: LinearConfig::new(input_dim, hidden_dim).init(device),
            fc2: LinearConfig::new(hidden_dim, embedding_dim).init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let hidden = activation::relu(self.fc1.forward(input));
        l2_normalize(self.fc2.forward(hidden))
    }
}

#[derive(Module, Debug)]
pub struct RetrievalModel<B: Backend> {
    pub pose_tower: RetrievalTower<B>,
    pub glyph_tower: RetrievalTower<B>,
    #[module(skip)]
    pub config: RetrievalModelConfig,
}

impl<B: Backend> RetrievalModel<B> {
    pub fn forward_pose(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.pose_tower.forward(input)
    }

    pub fn forward_glyph(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        self.glyph_tower.forward(input)
    }
}

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

pub fn save_retrieval_model<B: Backend>(
    model: &RetrievalModel<B>,
    checkpoint_dir: &Path,
    name: &str,
) -> Result<(), RetrievalError> {
    let recorder = DefaultRecorder::default();
    model
        .clone()
        .save_file(checkpoint_dir.join(name), &recorder)
        .map_err(RetrievalError::Recorder)
}

pub fn load_retrieval_model<B: Backend>(
    config: &RetrievalModelConfig,
    model_path: impl AsRef<Path>,
    device: &B::Device,
) -> Result<RetrievalModel<B>, RetrievalError> {
    let recorder = DefaultRecorder::default();
    let model_path = recorder_path_stem(model_path.as_ref());
    config
        .init(device)
        .load_file(model_path, &recorder, device)
        .map_err(RetrievalError::Recorder)
}

pub fn load_retrieval_model_config(
    path: impl AsRef<Path>,
) -> Result<RetrievalModelConfig, RetrievalError> {
    read_json_file(path)
}

pub fn encode_pose_features<B: Backend>(
    model: &RetrievalModel<B>,
    features: &[f32],
    device: &B::Device,
) -> Result<Vec<f32>, RetrievalError> {
    ensure_feature_dim(features.len(), model.config.input_dim)?;
    ensure_finite_values("pose feature", features)?;
    let input = Tensor::<B, 2>::from_data(
        TensorData::new(features.to_vec(), [1, model.config.input_dim]),
        device,
    );
    tensor_to_vec(model.forward_pose(input))
}

pub fn encode_glyph_features<B: Backend>(
    model: &RetrievalModel<B>,
    features: &[f32],
    device: &B::Device,
) -> Result<Vec<f32>, RetrievalError> {
    ensure_feature_dim(features.len(), model.config.input_dim)?;
    ensure_finite_values("glyph feature", features)?;
    let input = Tensor::<B, 2>::from_data(
        TensorData::new(features.to_vec(), [1, model.config.input_dim]),
        device,
    );
    tensor_to_vec(model.forward_glyph(input))
}

pub fn resolve_existing_data_root(data_root: &Path) -> Result<PathBuf, RetrievalError> {
    if data_root.is_dir() {
        return Ok(canonical_or_original(data_root));
    }
    if data_root == Path::new("data") {
        let parent = Path::new("..").join("data");
        if parent.is_dir() {
            return Ok(canonical_or_original(parent));
        }
    }
    Err(RetrievalError::InvalidData(format!(
        "data root does not exist: {}",
        data_root.display()
    )))
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
    feature_cache_dir: Option<&Path>,
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

pub(crate) fn extract_glyph_features_with_cache(
    path: &Path,
    feature_cache_dir: Option<&Path>,
) -> Result<Vec<f32>, RetrievalError> {
    cached_or_extract_features(feature_cache_dir, FeatureKind::Glyph, path, None, || {
        extract_glyph_features_from_path(path)
    })
}

fn extract_pose_features_with_cache(
    path: &Path,
    feature_cache_dir: Option<&Path>,
) -> Result<Vec<f32>, RetrievalError> {
    let pose_path = find_spinepose_json_for_image(path);
    cached_or_extract_features(
        feature_cache_dir,
        FeatureKind::Pose,
        path,
        pose_path.as_deref(),
        || DefaultPoseEstimator::default().estimate_pose_features_from_path(path),
    )
}

fn cached_or_extract_features<F>(
    feature_cache_dir: Option<&Path>,
    kind: FeatureKind,
    path: &Path,
    auxiliary_path: Option<&Path>,
    extract: F,
) -> Result<Vec<f32>, RetrievalError>
where
    F: FnOnce() -> Result<Vec<f32>, RetrievalError>,
{
    let Some(feature_cache_dir) = feature_cache_dir else {
        return extract();
    };

    let primary = CacheSourceFingerprint::from_path(path)?;
    let auxiliary = auxiliary_path
        .map(CacheSourceFingerprint::from_path)
        .transpose()?;
    let cache_path = feature_cache_dir
        .join(kind.cache_dir_name())
        .join(format!("{}.json", feature_cache_key(kind, path)));

    if let Ok(cached) = read_feature_cache(&cache_path)
        && cached.matches(kind, &primary, auxiliary.as_ref())
    {
        return Ok(cached.features);
    }

    let features = extract()?;
    ensure_feature_dim(features.len(), RETRIEVAL_FEATURE_DIM)?;
    ensure_finite_values(kind.feature_label(), &features)?;

    if let Some(parent) = cache_path.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json_file(
        &cache_path,
        &FeatureCacheEntry {
            version: FEATURE_CACHE_VERSION,
            kind: kind.as_str().to_string(),
            feature_dim: RETRIEVAL_FEATURE_DIM,
            primary,
            auxiliary,
            features: features.clone(),
        },
    )?;

    Ok(features)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum FeatureKind {
    Pose,
    Glyph,
}

impl FeatureKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Pose => "pose",
            Self::Glyph => "glyph",
        }
    }

    fn cache_dir_name(self) -> &'static str {
        self.as_str()
    }

    fn feature_label(self) -> &'static str {
        match self {
            Self::Pose => "pose feature",
            Self::Glyph => "glyph feature",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FeatureCacheEntry {
    version: u32,
    kind: String,
    feature_dim: usize,
    primary: CacheSourceFingerprint,
    auxiliary: Option<CacheSourceFingerprint>,
    features: Vec<f32>,
}

impl FeatureCacheEntry {
    fn matches(
        &self,
        kind: FeatureKind,
        primary: &CacheSourceFingerprint,
        auxiliary: Option<&CacheSourceFingerprint>,
    ) -> bool {
        self.version == FEATURE_CACHE_VERSION
            && self.kind == kind.as_str()
            && self.feature_dim == RETRIEVAL_FEATURE_DIM
            && &self.primary == primary
            && self.auxiliary.as_ref() == auxiliary
            && self.features.len() == RETRIEVAL_FEATURE_DIM
            && self.features.iter().all(|value| value.is_finite())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
struct CacheSourceFingerprint {
    path: String,
    len: u64,
    modified_secs: u64,
    modified_nanos: u32,
}

impl CacheSourceFingerprint {
    fn from_path(path: &Path) -> Result<Self, RetrievalError> {
        let metadata = fs::metadata(path)?;
        let modified = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .unwrap_or_default();
        Ok(Self {
            path: canonical_or_original(path).to_string_lossy().to_string(),
            len: metadata.len(),
            modified_secs: modified.as_secs(),
            modified_nanos: modified.subsec_nanos(),
        })
    }
}

fn read_feature_cache(path: &Path) -> Result<FeatureCacheEntry, RetrievalError> {
    read_json_file(path)
}

fn feature_cache_key(kind: FeatureKind, path: &Path) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    let path_text = canonical_or_original(path).to_string_lossy().to_string();
    for byte in kind.as_str().bytes().chain([0_u8]).chain(path_text.bytes()) {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
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

fn l2_normalize<B: Backend>(embedding: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = embedding.clone().square().sum_dim(1).sqrt().clamp_min(1e-6);
    embedding / norm
}

fn extract_shape_features(image: &DynamicImage) -> Vec<f32> {
    let rgba = image.to_rgba8();
    let (width, height) = image.dimensions();
    if width == 0 || height == 0 {
        return vec![0.0; RETRIEVAL_FEATURE_DIM];
    }

    let bg_luma = border_luma(&rgba, width, height);
    let mut masses = Vec::with_capacity((width * height) as usize);
    let mut max_mass = 0.0_f32;
    let mut total_mass = 0.0_f32;

    for pixel in rgba.pixels() {
        let alpha = pixel[3] as f32 / 255.0;
        let luma = luma(pixel[0], pixel[1], pixel[2]);
        let contrast = (luma - bg_luma).abs();
        let darkness = (bg_luma - luma).max(0.0);
        let mass = alpha * (0.65 * contrast + 0.35 * darkness);
        max_mass = max_mass.max(mass);
        total_mass += mass;
        masses.push(mass);
    }

    if total_mass <= 1e-6 || max_mass <= 1e-6 {
        return vec![0.0; RETRIEVAL_FEATURE_DIM];
    }

    let threshold = (max_mass * 0.20).max(total_mass / masses.len() as f32 * 0.75);
    let bbox = foreground_bbox(&masses, width, height, threshold).unwrap_or((
        0,
        0,
        width.saturating_sub(1),
        height.saturating_sub(1),
    ));
    let (min_x, min_y, max_x, max_y) = bbox;
    let bbox_width = (max_x.saturating_sub(min_x) + 1).max(1) as f32;
    let bbox_height = (max_y.saturating_sub(min_y) + 1).max(1) as f32;
    let center_x = min_x as f32 + bbox_width * 0.5;
    let center_y = min_y as f32 + bbox_height * 0.5;
    let scale = bbox_width.max(bbox_height).max(1.0);

    let mut features = Vec::with_capacity(RETRIEVAL_FEATURE_DIM);
    let mut points_written = 0usize;
    for band in 0..FEATURE_BANDS {
        let band_top = min_y as f32 + bbox_height * band as f32 / FEATURE_BANDS as f32;
        let band_bottom = min_y as f32 + bbox_height * (band + 1) as f32 / FEATURE_BANDS as f32;
        let mut band_mass = 0.0;
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;
        let mut left_x = f32::MAX;
        let mut left_y = 0.0;
        let mut right_x = f32::MIN;
        let mut right_y = 0.0;

        for y in min_y..=max_y {
            let yf = y as f32 + 0.5;
            if yf < band_top || yf >= band_bottom {
                continue;
            }
            for x in min_x..=max_x {
                let mass = masses[(y * width + x) as usize];
                if mass < threshold * 0.5 {
                    continue;
                }
                let xf = x as f32 + 0.5;
                band_mass += mass;
                weighted_x += xf * mass;
                weighted_y += yf * mass;
                if xf < left_x {
                    left_x = xf;
                    left_y = yf;
                }
                if xf > right_x {
                    right_x = xf;
                    right_y = yf;
                }
            }
        }

        let confidence = (band_mass / (total_mass / FEATURE_BANDS as f32 + 1e-6)).clamp(0.0, 1.0);
        let fallback_y = (band_top + band_bottom) * 0.5;
        let (center_band_x, center_band_y) = if band_mass > 1e-6 {
            (weighted_x / band_mass, weighted_y / band_mass)
        } else {
            (center_x, fallback_y)
        };

        let points = if band_mass > 1e-6 {
            [
                (left_x, left_y, confidence),
                (center_band_x, center_band_y, confidence),
                (right_x, right_y, confidence),
            ]
        } else {
            [
                (center_x, fallback_y, 0.0),
                (center_x, fallback_y, 0.0),
                (center_x, fallback_y, 0.0),
            ]
        };

        for (x, y, confidence) in points.into_iter().take(FEATURE_POINTS_PER_BAND) {
            if points_written >= RETRIEVAL_KEYPOINTS {
                break;
            }
            features.push(((x - center_x) / scale).clamp(-1.5, 1.5));
            features.push(((y - center_y) / scale).clamp(-1.5, 1.5));
            features.push(confidence);
            points_written += 1;
        }
    }

    debug_assert_eq!(features.len(), RETRIEVAL_FEATURE_DIM);
    features
}

fn border_luma(image: &image::RgbaImage, width: u32, height: u32) -> f32 {
    let mut total = 0.0_f32;
    let mut count = 0.0_f32;

    for x in 0..width {
        for y in [0, height.saturating_sub(1)] {
            let pixel = image.get_pixel(x, y);
            total += luma(pixel[0], pixel[1], pixel[2]);
            count += 1.0;
        }
    }
    for y in 1..height.saturating_sub(1) {
        for x in [0, width.saturating_sub(1)] {
            let pixel = image.get_pixel(x, y);
            total += luma(pixel[0], pixel[1], pixel[2]);
            count += 1.0;
        }
    }

    if count > 0.0 { total / count } else { 1.0 }
}

fn foreground_bbox(
    masses: &[f32],
    width: u32,
    height: u32,
    threshold: f32,
) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut found = false;

    for y in 0..height {
        for x in 0..width {
            if masses[(y * width + x) as usize] < threshold {
                continue;
            }
            found = true;
            min_x = min_x.min(x);
            min_y = min_y.min(y);
            max_x = max_x.max(x);
            max_y = max_y.max(y);
        }
    }

    found.then_some((min_x, min_y, max_x, max_y))
}

fn luma(red: u8, green: u8, blue: u8) -> f32 {
    (0.2126 * red as f32 + 0.7152 * green as f32 + 0.0722 * blue as f32) / 255.0
}

fn tensor_to_vec<B: Backend>(tensor: Tensor<B, 2>) -> Result<Vec<f32>, RetrievalError> {
    tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|error| RetrievalError::Tensor(format!("{error:?}")))
}

fn ensure_feature_dim(actual: usize, expected: usize) -> Result<(), RetrievalError> {
    if actual == expected {
        Ok(())
    } else {
        Err(RetrievalError::InvalidData(format!(
            "feature dimension mismatch: expected {expected}, got {actual}"
        )))
    }
}

fn ensure_finite_values(label: &str, values: &[f32]) -> Result<(), RetrievalError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(RetrievalError::InvalidData(format!(
            "{label} at index {index} is not finite"
        )))
    } else {
        Ok(())
    }
}

fn is_png(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
}

fn canonical_or_original(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

struct GlyphIdMetadata {
    codepoint: Option<String>,
    character: Option<String>,
}

fn parse_glyph_id(id: &str) -> GlyphIdMetadata {
    let codepoint = id
        .split("_u")
        .nth(1)
        .and_then(|rest| rest.split('_').next())
        .map(str::to_owned);
    let character = codepoint
        .as_deref()
        .and_then(|hex| u32::from_str_radix(hex, 16).ok())
        .and_then(char::from_u32)
        .map(|ch| ch.to_string());
    GlyphIdMetadata {
        codepoint: codepoint.map(|value| format!("U+{}", value.to_uppercase())),
        character,
    }
}

fn recorder_path_stem(path: &Path) -> PathBuf {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mpk"))
    {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

fn write_json_file<T: Serialize>(path: impl AsRef<Path>, value: &T) -> Result<(), RetrievalError> {
    let json = json::to_string_pretty(value)?;
    fs::write(path, json)?;
    Ok(())
}

fn read_json_file<T: DeserializeOwned>(path: impl AsRef<Path>) -> Result<T, RetrievalError> {
    let mut contents = fs::read(path)?;
    Ok(json::from_slice(&mut contents)?)
}

extern crate ann as burn;
