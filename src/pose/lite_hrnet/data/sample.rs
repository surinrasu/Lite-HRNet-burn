use std::path::PathBuf;

use ann::{
    prelude::Backend,
    tensor::{Tensor, TensorData},
};

use super::super::train::PoseBatch;

#[derive(Clone, Debug)]
pub struct PoseDataConfig {
    pub input_height: usize,
    pub input_width: usize,
    pub heatmap_height: usize,
    pub heatmap_width: usize,
    pub num_joints: usize,
    pub sigma: f32,
    pub bbox_padding: f32,
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl PoseDataConfig {
    pub fn coco_256x192(num_joints: usize) -> Self {
        Self {
            input_height: 256,
            input_width: 192,
            heatmap_height: 64,
            heatmap_width: 48,
            num_joints,
            sigma: 2.0,
            bbox_padding: 1.25,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
        }
    }

    pub fn from_input(input_height: usize, input_width: usize, num_joints: usize) -> Self {
        Self {
            input_height,
            input_width,
            heatmap_height: input_height / 4,
            heatmap_width: input_width / 4,
            ..Self::coco_256x192(num_joints)
        }
    }
}

#[derive(Clone, Debug)]
pub struct PoseSample {
    pub image_path: PathBuf,
    pub image_width: u32,
    pub image_height: u32,
    pub bbox: [f32; 4],
    pub keypoints: Vec<[f32; 3]>,
}

#[derive(Clone, Debug)]
pub struct PoseTensorSample {
    pub image: Vec<f32>,
    pub target: Vec<f32>,
    pub target_weight: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct PoseTensorBatch {
    pub images: Vec<f32>,
    pub targets: Vec<f32>,
    pub target_weight: Vec<f32>,
    pub batch_size: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub heatmap_height: usize,
    pub heatmap_width: usize,
    pub num_joints: usize,
}

impl PoseTensorBatch {
    pub fn len(&self) -> usize {
        self.batch_size
    }

    pub fn is_empty(&self) -> bool {
        self.batch_size == 0
    }

    pub fn into_pose_batch<B: Backend>(self, device: &B::Device) -> PoseBatch<B> {
        PoseBatch {
            images: Tensor::from_data(
                TensorData::new(
                    self.images,
                    [self.batch_size, 3, self.input_height, self.input_width],
                ),
                device,
            ),
            targets: Tensor::from_data(
                TensorData::new(
                    self.targets,
                    [
                        self.batch_size,
                        self.num_joints,
                        self.heatmap_height,
                        self.heatmap_width,
                    ],
                ),
                device,
            ),
            target_weight: Tensor::from_data(
                TensorData::new(self.target_weight, [self.batch_size, self.num_joints, 1]),
                device,
            ),
        }
    }
}
