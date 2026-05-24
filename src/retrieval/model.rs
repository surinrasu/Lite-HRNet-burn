use std::path::Path;

use ann::{
    module::Module,
    nn::{Linear, LinearConfig},
    record::DefaultRecorder,
    tensor::{Tensor, TensorData, activation, backend::Backend},
};
use serde::{Deserialize, Serialize};

use super::{
    DEFAULT_RETRIEVAL_EMBEDDING_DIM, DEFAULT_RETRIEVAL_HIDDEN_DIM, RETRIEVAL_FEATURE_DIM,
    RetrievalError, ensure_feature_dim, ensure_finite_values, read_json_file, recorder_path_stem,
};

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

fn l2_normalize<B: Backend>(embedding: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = embedding.clone().square().sum_dim(1).sqrt().clamp_min(1e-6);
    embedding / norm
}

fn tensor_to_vec<B: Backend>(tensor: Tensor<B, 2>) -> Result<Vec<f32>, RetrievalError> {
    tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|error| RetrievalError::Tensor(format!("{error:?}")))
}
