use std::{collections::HashMap, fs, path::Path};

use ann::prelude::Backend;
use image::ImageReader;
use rayon::prelude::*;
use serde::Deserialize;

use super::super::train::PoseBatch;
use super::{
    PoseDataConfig, PoseDataError, PoseSample, PoseTensorBatch, PoseTensorSample,
    heatmap::generate_heatmaps,
    spinepose::{load_spinepose_people_by_image, match_spinepose_person},
    transform::{CropWindow, crop_and_normalize},
};

#[derive(Clone, Debug)]
pub struct CocoPoseDataset {
    samples: Vec<PoseSample>,
    config: PoseDataConfig,
}

impl CocoPoseDataset {
    pub fn from_coco(
        annotation_path: impl AsRef<Path>,
        image_root: impl AsRef<Path>,
        config: PoseDataConfig,
    ) -> Result<Self, PoseDataError> {
        Self::from_coco_with_pose_source(annotation_path, image_root, None, config)
    }

    pub fn from_coco_with_spinepose(
        annotation_path: impl AsRef<Path>,
        image_root: impl AsRef<Path>,
        spinepose_root: impl AsRef<Path>,
        config: PoseDataConfig,
    ) -> Result<Self, PoseDataError> {
        Self::from_coco_with_pose_source(
            annotation_path,
            image_root,
            Some(spinepose_root.as_ref()),
            config,
        )
    }

    fn from_coco_with_pose_source(
        annotation_path: impl AsRef<Path>,
        image_root: impl AsRef<Path>,
        spinepose_root: Option<&Path>,
        config: PoseDataConfig,
    ) -> Result<Self, PoseDataError> {
        let annotation_path = annotation_path.as_ref();
        let image_root = image_root.as_ref();
        let mut contents = fs::read(annotation_path)?;
        let annotations: CocoRoot = json::from_slice(&mut contents)?;

        let images = annotations
            .images
            .into_iter()
            .map(|image| (image.id, image))
            .collect::<HashMap<_, _>>();

        let spineposes = spinepose_root
            .map(|root| {
                load_spinepose_people_by_image(
                    root,
                    images
                        .iter()
                        .map(|(image_id, image)| (*image_id, image.file_name.as_str())),
                )
            })
            .transpose()?;

        let mut samples = Vec::new();
        for annotation in annotations.annotations {
            if annotation.iscrowd.unwrap_or(0) != 0 {
                continue;
            }
            if annotation.category_id.is_some_and(|category| category != 1) {
                continue;
            }
            if annotation.bbox.len() < 4 {
                continue;
            }

            let Some(image) = images.get(&annotation.image_id) else {
                continue;
            };

            let bbox = [
                annotation.bbox[0],
                annotation.bbox[1],
                annotation.bbox[2],
                annotation.bbox[3],
            ];
            if bbox[2] <= 1.0 || bbox[3] <= 1.0 {
                continue;
            }

            let keypoints = if let Some(spineposes) = &spineposes {
                let Some(people) = spineposes.get(&annotation.image_id) else {
                    continue;
                };
                let Some(keypoints) = match_spinepose_person(people, bbox, config.num_joints)
                else {
                    continue;
                };
                keypoints
            } else {
                let Some(keypoints) = coco_annotation_keypoints(&annotation, config.num_joints)
                else {
                    continue;
                };
                keypoints
            };

            samples.push(PoseSample {
                image_path: image_root.join(&image.file_name),
                image_width: image.width.unwrap_or(0),
                image_height: image.height.unwrap_or(0),
                bbox,
                keypoints,
            });
        }

        if samples.is_empty() {
            return Err(PoseDataError::InvalidDataset(format!(
                "no valid pose samples found in {}",
                annotation_path.display()
            )));
        }

        Ok(Self { samples, config })
    }

    pub fn from_samples(
        samples: Vec<PoseSample>,
        config: PoseDataConfig,
    ) -> Result<Self, PoseDataError> {
        if samples.is_empty() {
            return Err(PoseDataError::InvalidDataset(
                "dataset requires at least one sample".to_string(),
            ));
        }
        Ok(Self { samples, config })
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn config(&self) -> &PoseDataConfig {
        &self.config
    }

    pub fn samples(&self) -> &[PoseSample] {
        &self.samples
    }

    pub fn limited(&self, max_samples: Option<usize>) -> Self {
        match max_samples {
            Some(max_samples) => Self {
                samples: self.samples.iter().take(max_samples).cloned().collect(),
                config: self.config.clone(),
            },
            None => self.clone(),
        }
    }

    pub fn load_tensor_sample(&self, index: usize) -> Result<PoseTensorSample, PoseDataError> {
        let sample = self.samples.get(index).ok_or_else(|| {
            PoseDataError::InvalidDataset(format!("sample index {index} out of range"))
        })?;
        let image = ImageReader::open(&sample.image_path)?.decode()?.to_rgb8();
        let crop = CropWindow::from_bbox(sample.bbox, &self.config);

        let image_tensor = crop_and_normalize(&image, crop, &self.config);
        let (target, target_weight) = generate_heatmaps(sample, crop, &self.config);

        Ok(PoseTensorSample {
            image: image_tensor,
            target,
            target_weight,
        })
    }

    pub fn batch<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> Result<PoseBatch<B>, PoseDataError> {
        Ok(self.load_tensor_batch(indices)?.into_pose_batch(device))
    }

    pub fn load_tensor_batch(&self, indices: &[usize]) -> Result<PoseTensorBatch, PoseDataError> {
        if indices.is_empty() {
            return Err(PoseDataError::InvalidDataset(
                "batch requires at least one sample".to_string(),
            ));
        }

        let image_len = 3 * self.config.input_height * self.config.input_width;
        let target_len =
            self.config.num_joints * self.config.heatmap_height * self.config.heatmap_width;
        let weight_len = self.config.num_joints;

        let mut images = Vec::with_capacity(indices.len() * image_len);
        let mut targets = Vec::with_capacity(indices.len() * target_len);
        let mut weights = Vec::with_capacity(indices.len() * weight_len);

        let samples = indices
            .par_iter()
            .map(|&index| self.load_tensor_sample(index))
            .collect::<Vec<_>>();

        for sample in samples {
            let sample = sample?;
            images.extend(sample.image);
            targets.extend(sample.target);
            weights.extend(sample.target_weight);
        }

        Ok(PoseTensorBatch {
            images,
            targets,
            target_weight: weights,
            batch_size: indices.len(),
            input_height: self.config.input_height,
            input_width: self.config.input_width,
            heatmap_height: self.config.heatmap_height,
            heatmap_width: self.config.heatmap_width,
            num_joints: self.config.num_joints,
        })
    }
}

fn coco_annotation_keypoints(
    annotation: &CocoAnnotation,
    num_joints: usize,
) -> Option<Vec<[f32; 3]>> {
    if annotation.keypoints.len() < num_joints * 3 {
        return None;
    }
    if annotation.num_keypoints.unwrap_or(0) == 0 {
        let labeled = annotation
            .keypoints
            .chunks_exact(3)
            .take(num_joints)
            .any(|keypoint| keypoint[2] > 0.0);
        if !labeled {
            return None;
        }
    }

    Some(
        annotation
            .keypoints
            .chunks_exact(3)
            .take(num_joints)
            .map(|keypoint| [keypoint[0], keypoint[1], keypoint[2]])
            .collect(),
    )
}

#[derive(Debug, Deserialize)]
struct CocoRoot {
    images: Vec<CocoImage>,
    annotations: Vec<CocoAnnotation>,
}

#[derive(Debug, Deserialize)]
struct CocoImage {
    id: u64,
    file_name: String,
    width: Option<u32>,
    height: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct CocoAnnotation {
    image_id: u64,
    category_id: Option<u32>,
    bbox: Vec<f32>,
    keypoints: Vec<f32>,
    num_keypoints: Option<u32>,
    iscrowd: Option<u32>,
}
