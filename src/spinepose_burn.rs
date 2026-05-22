use std::{
    env,
    path::Path,
    sync::{Mutex, OnceLock},
};

use ann::{
    backend::Flex,
    tensor::{Tensor, TensorData, backend::BackendTypes},
};
use image::DynamicImage;

use crate::{RetrievalError, SPINEPOSE_KEYPOINTS};

const OUT_DIR_NAME: &str = "spinepose";
const DETECTOR_BPK: &str = concat!(env!("OUT_DIR"), "/spinepose/rfdetr_m_v142_576x576.bpk");
const POSE_BPK: &str = concat!(
    env!("OUT_DIR"),
    "/spinepose/spinepose-l_32xb256-10e_simspine-256x192.bpk"
);

const RFDETR_WIDTH: usize = 576;
const RFDETR_HEIGHT: usize = 576;
const RFDETR_SCORE_THRESHOLD: f32 = 0.3;
const RFDETR_NUM_SELECT: usize = 300;
const RFDETR_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const RFDETR_STD: [f32; 3] = [0.229, 0.224, 0.225];

const POSE_WIDTH: usize = 192;
const POSE_HEIGHT: usize = 256;
const POSE_PADDING: f32 = 1.25;
const POSE_SIMCC_SPLIT_RATIO: f32 = 2.0;
const POSE_MEAN: [f32; 3] = [123.675, 116.28, 103.53];
const POSE_STD: [f32; 3] = [58.395, 57.12, 57.375];

#[allow(dead_code, unused_imports, unused_mut, unused_variables, clippy::all)]
mod generated {
    pub mod rfdetr_m_v142_576x576 {
        include!(concat!(
            env!("OUT_DIR"),
            "/spinepose/rfdetr_m_v142_576x576.rs"
        ));
    }

    pub mod spinepose_l_32xb256_10e_simspine_256x192 {
        include!(concat!(
            env!("OUT_DIR"),
            "/spinepose/spinepose-l_32xb256-10e_simspine-256x192.rs"
        ));
    }
}

type BurnBackend = Flex;
type BurnDevice = <BurnBackend as BackendTypes>::Device;
type DetectorModel = generated::rfdetr_m_v142_576x576::Model<BurnBackend>;
type PoseModel = generated::spinepose_l_32xb256_10e_simspine_256x192::Model<BurnBackend>;

static RUNTIME: OnceLock<Result<Mutex<BurnSpinePoseRuntime>, String>> = OnceLock::new();

pub(crate) fn estimate_people(image: &DynamicImage) -> Result<Vec<Vec<[f32; 3]>>, RetrievalError> {
    validate_runtime_config()?;
    let image = BgrImage::from_dynamic(image);
    let runtime = RUNTIME.get_or_init(|| {
        BurnSpinePoseRuntime::new()
            .map(Mutex::new)
            .map_err(|error| error.to_string())
    });
    let runtime = runtime
        .as_ref()
        .map_err(|message| RetrievalError::InvalidData(message.clone()))?;
    let mut runtime = runtime
        .lock()
        .map_err(|_| RetrievalError::InvalidData("SpinePose Burn runtime lock poisoned".into()))?;
    runtime.estimate_people(&image)
}

struct BurnSpinePoseRuntime {
    detector: DetectorModel,
    pose: PoseModel,
    device: BurnDevice,
}

impl BurnSpinePoseRuntime {
    fn new() -> Result<Self, RetrievalError> {
        for path in [DETECTOR_BPK, POSE_BPK] {
            if !Path::new(path).is_file() {
                return Err(RetrievalError::InvalidData(format!(
                    "generated SpinePose burnpack is missing at {path}; rerun cargo build so build.rs can generate {OUT_DIR_NAME} models"
                )));
            }
        }

        let device = BurnDevice::default();
        Ok(Self {
            detector: DetectorModel::from_file(DETECTOR_BPK, &device),
            pose: PoseModel::from_file(POSE_BPK, &device),
            device,
        })
    }

    fn estimate_people(&mut self, image: &BgrImage) -> Result<Vec<Vec<[f32; 3]>>, RetrievalError> {
        let boxes = self.detect(image)?;
        let mut people = Vec::with_capacity(boxes.len());
        for bbox in boxes {
            people.push(self.estimate_bbox(image, bbox)?);
        }
        Ok(people)
    }

    fn detect(&self, image: &BgrImage) -> Result<Vec<BBox>, RetrievalError> {
        let input = rfdetr_input(image, &self.device);
        let (boxes, logits) = self.detector.forward(input);
        let (boxes, box_shape) = tensor3_to_vec(boxes)?;
        let (logits, logit_shape) = tensor3_to_vec(logits)?;

        if box_shape[0] != 1 || box_shape[2] != 4 {
            return Err(RetrievalError::InvalidData(format!(
                "RF-DETR boxes shape {:?} is unsupported",
                box_shape
            )));
        }
        if logit_shape[0] != 1 || logit_shape[1] != box_shape[1] || logit_shape[2] < 2 {
            return Err(RetrievalError::InvalidData(format!(
                "RF-DETR logits shape {:?} is unsupported for boxes {:?}",
                logit_shape, box_shape
            )));
        }

        let num_boxes = box_shape[1];
        let num_classes = logit_shape[2];
        let mut ranked = Vec::with_capacity(num_boxes);
        for box_index in 0..num_boxes {
            let person_logit = logits[box_index * num_classes + 1];
            ranked.push((sigmoid(person_logit), box_index));
        }
        ranked.sort_by(|left, right| right.0.total_cmp(&left.0));

        let image_width = image.width as f32;
        let image_height = image.height as f32;
        let mut detections = Vec::new();
        for (score, box_index) in ranked.into_iter().take(RFDETR_NUM_SELECT) {
            if score <= RFDETR_SCORE_THRESHOLD {
                continue;
            }

            let offset = box_index * 4;
            let cx = boxes[offset];
            let cy = boxes[offset + 1];
            let width = boxes[offset + 2].max(0.0);
            let height = boxes[offset + 3].max(0.0);
            let bbox = BBox {
                x1: (cx - width * 0.5) * image_width,
                y1: (cy - height * 0.5) * image_height,
                x2: (cx + width * 0.5) * image_width,
                y2: (cy + height * 0.5) * image_height,
            };
            if bbox.is_valid() {
                detections.push(bbox);
            }
        }

        Ok(detections)
    }

    fn estimate_bbox(&self, image: &BgrImage, bbox: BBox) -> Result<Vec<[f32; 3]>, RetrievalError> {
        let (input, center, scale) = pose_input(image, bbox, &self.device);
        let (simcc_x, simcc_y) = self.pose.forward(input);
        let (simcc_x, simcc_x_shape) = tensor3_to_vec(simcc_x)?;
        let (simcc_y, simcc_y_shape) = tensor3_to_vec(simcc_y)?;

        if simcc_x_shape[0] != 1 || simcc_y_shape[0] != 1 {
            return Err(RetrievalError::InvalidData(format!(
                "SpinePose batch shapes {:?} and {:?} are unsupported",
                simcc_x_shape, simcc_y_shape
            )));
        }
        if simcc_x_shape[1] != SPINEPOSE_KEYPOINTS || simcc_y_shape[1] != SPINEPOSE_KEYPOINTS {
            return Err(RetrievalError::InvalidData(format!(
                "SpinePose returned {:?}/{:?} keypoints, expected {SPINEPOSE_KEYPOINTS}",
                simcc_x_shape, simcc_y_shape
            )));
        }

        let wx = simcc_x_shape[2];
        let wy = simcc_y_shape[2];
        let mut keypoints = Vec::with_capacity(SPINEPOSE_KEYPOINTS);
        for keypoint_index in 0..SPINEPOSE_KEYPOINTS {
            let x_base = keypoint_index * wx;
            let y_base = keypoint_index * wy;
            let (x_index, x_score) = max_index(&simcc_x[x_base..x_base + wx]);
            let (y_index, y_score) = max_index(&simcc_y[y_base..y_base + wy]);
            let score = 0.5 * (x_score + y_score);

            let (x, y) = if score > 0.0 && score.is_finite() {
                let x = x_index as f32 / POSE_SIMCC_SPLIT_RATIO / POSE_WIDTH as f32 * scale.0
                    + center.0
                    - scale.0 * 0.5;
                let y = y_index as f32 / POSE_SIMCC_SPLIT_RATIO / POSE_HEIGHT as f32 * scale.1
                    + center.1
                    - scale.1 * 0.5;
                (x, y)
            } else {
                (-1.0, -1.0)
            };
            keypoints.push([x, y, score.clamp(0.0, 1.0)]);
        }

        Ok(keypoints)
    }
}

#[derive(Clone, Copy, Debug)]
struct BBox {
    x1: f32,
    y1: f32,
    x2: f32,
    y2: f32,
}

impl BBox {
    fn is_valid(self) -> bool {
        self.x1.is_finite()
            && self.y1.is_finite()
            && self.x2.is_finite()
            && self.y2.is_finite()
            && self.x2 > self.x1
            && self.y2 > self.y1
    }

    fn center_scale(self) -> ((f32, f32), (f32, f32)) {
        let center = ((self.x1 + self.x2) * 0.5, (self.y1 + self.y2) * 0.5);
        let mut scale = (
            (self.x2 - self.x1) * POSE_PADDING,
            (self.y2 - self.y1) * POSE_PADDING,
        );
        let aspect_ratio = POSE_WIDTH as f32 / POSE_HEIGHT as f32;
        if scale.0 > scale.1 * aspect_ratio {
            scale.1 = scale.0 / aspect_ratio;
        } else {
            scale.0 = scale.1 * aspect_ratio;
        }
        (center, scale)
    }
}

struct BgrImage {
    width: usize,
    height: usize,
    pixels: Vec<[f32; 3]>,
}

impl BgrImage {
    fn from_dynamic(image: &DynamicImage) -> Self {
        let rgb = image.to_rgb8();
        let width = rgb.width() as usize;
        let height = rgb.height() as usize;
        let pixels = rgb
            .pixels()
            .map(|pixel| [pixel[2] as f32, pixel[1] as f32, pixel[0] as f32])
            .collect();
        Self {
            width,
            height,
            pixels,
        }
    }

    fn pixel(&self, x: usize, y: usize) -> [f32; 3] {
        self.pixels[y * self.width + x]
    }

    fn sample_clamped(&self, x: f32, y: f32) -> [f32; 3] {
        let x = x.clamp(0.0, (self.width.saturating_sub(1)) as f32);
        let y = y.clamp(0.0, (self.height.saturating_sub(1)) as f32);
        self.sample_bilinear(x, y, true)
    }

    fn sample_zero(&self, x: f32, y: f32) -> [f32; 3] {
        self.sample_bilinear(x, y, false)
    }

    fn sample_bilinear(&self, x: f32, y: f32, clamp: bool) -> [f32; 3] {
        if self.width == 0 || self.height == 0 {
            return [0.0; 3];
        }

        let x0 = x.floor() as i32;
        let y0 = y.floor() as i32;
        let wx = x - x0 as f32;
        let wy = y - y0 as f32;

        let mut output = [0.0; 3];
        for (dy, y_weight) in [(0, 1.0 - wy), (1, wy)] {
            for (dx, x_weight) in [(0, 1.0 - wx), (1, wx)] {
                let mut sx = x0 + dx;
                let mut sy = y0 + dy;
                if clamp {
                    sx = sx.clamp(0, self.width as i32 - 1);
                    sy = sy.clamp(0, self.height as i32 - 1);
                } else if sx < 0 || sy < 0 || sx >= self.width as i32 || sy >= self.height as i32 {
                    continue;
                }

                let pixel = self.pixel(sx as usize, sy as usize);
                let weight = x_weight * y_weight;
                for channel in 0..3 {
                    output[channel] += pixel[channel] * weight;
                }
            }
        }
        output
    }
}

fn rfdetr_input(image: &BgrImage, device: &BurnDevice) -> Tensor<BurnBackend, 4> {
    let resized = resize_bilinear(image, RFDETR_WIDTH, RFDETR_HEIGHT);
    let mut data = vec![0.0; 3 * RFDETR_HEIGHT * RFDETR_WIDTH];
    for y in 0..RFDETR_HEIGHT {
        for x in 0..RFDETR_WIDTH {
            let pixel = resized[y * RFDETR_WIDTH + x];
            for channel in 0..3 {
                let value = (pixel[channel] / 255.0 - RFDETR_MEAN[channel]) / RFDETR_STD[channel];
                data[channel * RFDETR_HEIGHT * RFDETR_WIDTH + y * RFDETR_WIDTH + x] = value;
            }
        }
    }

    Tensor::from_data(
        TensorData::new(data, [1, 3, RFDETR_HEIGHT, RFDETR_WIDTH]),
        device,
    )
}

fn resize_bilinear(image: &BgrImage, width: usize, height: usize) -> Vec<[f32; 3]> {
    let mut output = vec![[0.0; 3]; width * height];
    let scale_x = image.width as f32 / width as f32;
    let scale_y = image.height as f32 / height as f32;

    for y in 0..height {
        let src_y = (y as f32 + 0.5) * scale_y - 0.5;
        for x in 0..width {
            let src_x = (x as f32 + 0.5) * scale_x - 0.5;
            output[y * width + x] = image.sample_clamped(src_x, src_y);
        }
    }

    output
}

fn pose_input(
    image: &BgrImage,
    bbox: BBox,
    device: &BurnDevice,
) -> (Tensor<BurnBackend, 4>, (f32, f32), (f32, f32)) {
    let (center, scale) = bbox.center_scale();
    let mut data = vec![0.0; 3 * POSE_HEIGHT * POSE_WIDTH];
    for y in 0..POSE_HEIGHT {
        let src_y = center.1 + (y as f32 - POSE_HEIGHT as f32 * 0.5) * scale.1 / POSE_HEIGHT as f32;
        for x in 0..POSE_WIDTH {
            let src_x =
                center.0 + (x as f32 - POSE_WIDTH as f32 * 0.5) * scale.0 / POSE_WIDTH as f32;
            let pixel = image.sample_zero(src_x, src_y);
            for channel in 0..3 {
                let value = (pixel[channel] - POSE_MEAN[channel]) / POSE_STD[channel];
                data[channel * POSE_HEIGHT * POSE_WIDTH + y * POSE_WIDTH + x] = value;
            }
        }
    }

    (
        Tensor::from_data(
            TensorData::new(data, [1, 3, POSE_HEIGHT, POSE_WIDTH]),
            device,
        ),
        center,
        scale,
    )
}

fn tensor3_to_vec(
    tensor: Tensor<BurnBackend, 3>,
) -> Result<(Vec<f32>, [usize; 3]), RetrievalError> {
    let shape = tensor.shape().dims::<3>();
    let data = tensor
        .into_data()
        .into_vec::<f32>()
        .map_err(|error| RetrievalError::Tensor(format!("{error:?}")))?;
    Ok((data, shape))
}

fn max_index(values: &[f32]) -> (usize, f32) {
    values
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap_or((0, f32::NEG_INFINITY))
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn validate_runtime_config() -> Result<(), RetrievalError> {
    require_env_value("SPINEPOSE_MODE", "large", &["large"])?;
    require_env_value("SPINEPOSE_DETECTOR", "rfdetr", &["rfdetr"])?;
    require_env_value("SPINEPOSE_MODEL_VERSION", "v2", &["v2", "latest"])?;
    Ok(())
}

fn require_env_value(name: &str, default: &str, supported: &[&str]) -> Result<(), RetrievalError> {
    let value = env::var(name).unwrap_or_else(|_| default.to_string());
    if supported
        .iter()
        .any(|supported| value.eq_ignore_ascii_case(supported))
    {
        return Ok(());
    }

    Err(RetrievalError::InvalidData(format!(
        "{name}={value:?} is not supported by the Burn SpinePose runtime; supported values: {}",
        supported.join(", ")
    )))
}
