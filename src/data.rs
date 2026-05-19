use std::{
    collections::HashMap,
    error::Error,
    fmt::{Display, Formatter},
    fs,
    path::{Path, PathBuf},
};

use burn::tensor::{Tensor, TensorData, backend::AutodiffBackend};
use image::{ImageReader, RgbImage};
use serde::Deserialize;

use crate::train::PoseBatch;

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
        let annotation_path = annotation_path.as_ref();
        let image_root = image_root.as_ref();
        let mut contents = fs::read(annotation_path)?;
        let annotations: CocoRoot = json::from_slice(&mut contents)?;

        let images = annotations
            .images
            .into_iter()
            .map(|image| (image.id, image))
            .collect::<HashMap<_, _>>();

        let mut samples = Vec::new();
        for annotation in annotations.annotations {
            if annotation.iscrowd.unwrap_or(0) != 0 {
                continue;
            }
            if annotation.bbox.len() < 4 || annotation.keypoints.len() < config.num_joints * 3 {
                continue;
            }
            if annotation.num_keypoints.unwrap_or(0) == 0 {
                let labeled = annotation
                    .keypoints
                    .chunks_exact(3)
                    .take(config.num_joints)
                    .any(|keypoint| keypoint[2] > 0.0);
                if !labeled {
                    continue;
                }
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

            let keypoints = annotation
                .keypoints
                .chunks_exact(3)
                .take(config.num_joints)
                .map(|keypoint| [keypoint[0], keypoint[1], keypoint[2]])
                .collect::<Vec<_>>();

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

    pub fn batch<B: AutodiffBackend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> Result<PoseBatch<B>, PoseDataError> {
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

        for &index in indices {
            let sample = self.load_tensor_sample(index)?;
            images.extend(sample.image);
            targets.extend(sample.target);
            weights.extend(sample.target_weight);
        }

        Ok(PoseBatch {
            images: Tensor::from_data(
                TensorData::new(
                    images,
                    [
                        indices.len(),
                        3,
                        self.config.input_height,
                        self.config.input_width,
                    ],
                ),
                device,
            ),
            targets: Tensor::from_data(
                TensorData::new(
                    targets,
                    [
                        indices.len(),
                        self.config.num_joints,
                        self.config.heatmap_height,
                        self.config.heatmap_width,
                    ],
                ),
                device,
            ),
            target_weight: Tensor::from_data(
                TensorData::new(weights, [indices.len(), self.config.num_joints, 1]),
                device,
            ),
        })
    }
}

#[derive(Clone, Copy, Debug)]
struct CropWindow {
    left: f32,
    top: f32,
    width: f32,
    height: f32,
}

impl CropWindow {
    fn from_bbox(bbox: [f32; 4], config: &PoseDataConfig) -> Self {
        let center_x = bbox[0] + bbox[2] * 0.5;
        let center_y = bbox[1] + bbox[3] * 0.5;
        let aspect = config.input_width as f32 / config.input_height as f32;

        let mut width = bbox[2];
        let mut height = bbox[3];
        if width > aspect * height {
            height = width / aspect;
        } else if width < aspect * height {
            width = height * aspect;
        }

        width *= config.bbox_padding;
        height *= config.bbox_padding;

        Self {
            left: center_x - width * 0.5,
            top: center_y - height * 0.5,
            width,
            height,
        }
    }

    fn transform_point(&self, x: f32, y: f32, config: &PoseDataConfig) -> (f32, f32) {
        (
            (x - self.left) * config.input_width as f32 / self.width,
            (y - self.top) * config.input_height as f32 / self.height,
        )
    }
}

fn crop_and_normalize(image: &RgbImage, crop: CropWindow, config: &PoseDataConfig) -> Vec<f32> {
    let mut output = vec![0.0; 3 * config.input_height * config.input_width];

    for y in 0..config.input_height {
        for x in 0..config.input_width {
            let src_x = crop.left + (x as f32 + 0.5) * crop.width / config.input_width as f32 - 0.5;
            let src_y =
                crop.top + (y as f32 + 0.5) * crop.height / config.input_height as f32 - 0.5;
            let rgb = bilinear_rgb(image, src_x, src_y);

            for (channel, pixel) in rgb.iter().enumerate() {
                let value = (*pixel / 255.0 - config.mean[channel]) / config.std[channel];
                let offset =
                    channel * config.input_height * config.input_width + y * config.input_width + x;
                output[offset] = value;
            }
        }
    }

    output
}

fn bilinear_rgb(image: &RgbImage, x: f32, y: f32) -> [f32; 3] {
    let width = image.width() as i32;
    let height = image.height() as i32;
    if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return [0.0; 3];
    }

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;

    let p00 = image.get_pixel(x0 as u32, y0 as u32);
    let p10 = image.get_pixel(x1 as u32, y0 as u32);
    let p01 = image.get_pixel(x0 as u32, y1 as u32);
    let p11 = image.get_pixel(x1 as u32, y1 as u32);

    let mut output = [0.0; 3];
    for channel in 0..3 {
        let top = p00[channel] as f32 * (1.0 - wx) + p10[channel] as f32 * wx;
        let bottom = p01[channel] as f32 * (1.0 - wx) + p11[channel] as f32 * wx;
        output[channel] = top * (1.0 - wy) + bottom * wy;
    }
    output
}

fn generate_heatmaps(
    sample: &PoseSample,
    crop: CropWindow,
    config: &PoseDataConfig,
) -> (Vec<f32>, Vec<f32>) {
    let mut target =
        vec![0.0_f32; config.num_joints * config.heatmap_height * config.heatmap_width];
    let mut target_weight = vec![0.0_f32; config.num_joints];
    let radius = (config.sigma * 3.0).ceil() as i32;
    let denominator = 2.0 * config.sigma * config.sigma;

    for (joint_index, keypoint) in sample.keypoints.iter().take(config.num_joints).enumerate() {
        if keypoint[2] <= 0.0 {
            continue;
        }

        let (input_x, input_y) = crop.transform_point(keypoint[0], keypoint[1], config);
        let mu_x = input_x * config.heatmap_width as f32 / config.input_width as f32;
        let mu_y = input_y * config.heatmap_height as f32 / config.input_height as f32;

        if mu_x < 0.0
            || mu_y < 0.0
            || mu_x >= config.heatmap_width as f32
            || mu_y >= config.heatmap_height as f32
        {
            continue;
        }

        target_weight[joint_index] = 1.0;
        let center_x = mu_x.round() as i32;
        let center_y = mu_y.round() as i32;
        let left = (center_x - radius).max(0);
        let right = (center_x + radius).min(config.heatmap_width as i32 - 1);
        let top = (center_y - radius).max(0);
        let bottom = (center_y + radius).min(config.heatmap_height as i32 - 1);

        for y in top..=bottom {
            for x in left..=right {
                let dx = x as f32 - mu_x;
                let dy = y as f32 - mu_y;
                let value = (-(dx * dx + dy * dy) / denominator).exp();
                let offset = joint_index * config.heatmap_height * config.heatmap_width
                    + y as usize * config.heatmap_width
                    + x as usize;
                target[offset] = target[offset].max(value);
            }
        }
    }

    (target, target_weight)
}

#[derive(Debug)]
pub enum PoseDataError {
    Io(std::io::Error),
    Json(json::Error),
    Image(image::ImageError),
    InvalidDataset(String),
}

impl Display for PoseDataError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "io error: {error}"),
            Self::Json(error) => write!(formatter, "json error: {error}"),
            Self::Image(error) => write!(formatter, "image error: {error}"),
            Self::InvalidDataset(message) => write!(formatter, "invalid dataset: {message}"),
        }
    }
}

impl Error for PoseDataError {}

impl From<std::io::Error> for PoseDataError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<json::Error> for PoseDataError {
    fn from(value: json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<image::ImageError> for PoseDataError {
    fn from(value: image::ImageError) -> Self {
        Self::Image(value)
    }
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
    bbox: Vec<f32>,
    keypoints: Vec<f32>,
    num_keypoints: Option<u32>,
    iscrowd: Option<u32>,
}
