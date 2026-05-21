use std::{
    env, fs,
    io::{Cursor, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use image::{DynamicImage, ImageFormat};
use serde::Deserialize;

use crate::RetrievalError;

pub const SPINEPOSE_KEYPOINTS: usize = 37;
pub const SPINEPOSE_VALUES_PER_KEYPOINT: usize = 3;
pub const SPINEPOSE_FEATURE_DIM: usize = SPINEPOSE_KEYPOINTS * SPINEPOSE_VALUES_PER_KEYPOINT;

const MIN_KEYPOINT_CONFIDENCE: f32 = 0.05;
const RUNTIME_SCRIPT: &str = r#"
import json
import os
import sys

import cv2
import numpy as np
from spinepose import SpinePoseEstimator

image_arg = sys.argv[1]
if image_arg == "-":
    encoded = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
else:
    image = cv2.imread(image_arg, cv2.IMREAD_COLOR)
if image is None:
    raise SystemExit(f"cannot read image: {image_arg}")

estimator = SpinePoseEstimator(
    mode=os.environ.get("SPINEPOSE_MODE", "large"),
    backend=os.environ.get("SPINEPOSE_BACKEND", "onnxruntime"),
    device=os.environ.get("SPINEPOSE_DEVICE", "cpu"),
    detector=os.environ.get("SPINEPOSE_DETECTOR", "rfdetr"),
    model_version=os.environ.get("SPINEPOSE_MODEL_VERSION", "v2"),
)

keypoints, scores = estimator(image)
people = []
if len(keypoints) > 0:
    keypoints = np.asarray(keypoints)
    scores = np.asarray(scores)
    if keypoints.shape[1] != 37:
        raise SystemExit(f"expected 37 SpinePose keypoints, got {keypoints.shape[1]}")
    poses = np.concatenate([keypoints, scores[..., np.newaxis]], axis=-1)
    for person in poses:
        people.append({"pose_keypoints_2d": person.reshape(-1).tolist()})

print(json.dumps({"version": 1.0, "people": people}))
"#;

pub trait PoseFeatureEstimator {
    fn estimate_pose_features(&self, image: &DynamicImage) -> Result<Vec<f32>, RetrievalError>;
    fn estimate_pose_features_from_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>, RetrievalError>;
}

#[derive(Clone, Debug, Default)]
pub struct SpinePoseEstimator;

impl SpinePoseEstimator {
    pub fn estimate_pose_features_from_path(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<Vec<f32>, RetrievalError> {
        let path = path.as_ref();
        if let Some(pose_path) = find_spinepose_json_for_image(path) {
            return read_spinepose_features(pose_path);
        }

        let keypoints = run_spinepose(SpinePoseInput::Path(path))?;
        spinepose_keypoints_to_features(&keypoints)
    }

    pub fn estimate_pose_features_from_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<Vec<f32>, RetrievalError> {
        let keypoints = run_spinepose(SpinePoseInput::EncodedBytes(bytes))?;
        spinepose_keypoints_to_features(&keypoints)
    }
}

impl PoseFeatureEstimator for SpinePoseEstimator {
    fn estimate_pose_features(&self, image: &DynamicImage) -> Result<Vec<f32>, RetrievalError> {
        let mut encoded = Cursor::new(Vec::new());
        image.write_to(&mut encoded, ImageFormat::Png)?;
        self.estimate_pose_features_from_bytes(&encoded.into_inner())
    }

    fn estimate_pose_features_from_bytes(&self, bytes: &[u8]) -> Result<Vec<f32>, RetrievalError> {
        SpinePoseEstimator::estimate_pose_features_from_bytes(self, bytes)
    }
}

pub type DefaultPoseEstimator = SpinePoseEstimator;

pub fn estimate_pose_features_from_image(image: &DynamicImage) -> Result<Vec<f32>, RetrievalError> {
    DefaultPoseEstimator::default().estimate_pose_features(image)
}

pub fn estimate_pose_features_from_path(
    path: impl AsRef<Path>,
) -> Result<Vec<f32>, RetrievalError> {
    DefaultPoseEstimator::default().estimate_pose_features_from_path(path)
}

pub fn estimate_pose_features_from_bytes(bytes: &[u8]) -> Result<Vec<f32>, RetrievalError> {
    DefaultPoseEstimator::default().estimate_pose_features_from_bytes(bytes)
}

pub fn read_spinepose_features(path: impl AsRef<Path>) -> Result<Vec<f32>, RetrievalError> {
    let people = read_spinepose_people(path)?;
    let keypoints = best_spinepose_person(&people).ok_or_else(|| {
        RetrievalError::InvalidData("SpinePose JSON did not contain any people".to_string())
    })?;
    spinepose_keypoints_to_features(keypoints)
}

pub fn read_spinepose_people(path: impl AsRef<Path>) -> Result<Vec<Vec<[f32; 3]>>, RetrievalError> {
    let path = path.as_ref();
    let mut contents = fs::read(path)?;
    let root: OpenPoseRoot = json::from_slice(&mut contents)?;
    let mut people = Vec::with_capacity(root.people.len());

    for person in root.people {
        let expected = SPINEPOSE_FEATURE_DIM;
        if person.pose_keypoints_2d.len() < expected {
            return Err(RetrievalError::InvalidData(format!(
                "SpinePose JSON {} has {} values for a person, expected at least {expected}",
                path.display(),
                person.pose_keypoints_2d.len()
            )));
        }

        people.push(
            person
                .pose_keypoints_2d
                .chunks_exact(SPINEPOSE_VALUES_PER_KEYPOINT)
                .take(SPINEPOSE_KEYPOINTS)
                .map(|keypoint| [keypoint[0], keypoint[1], keypoint[2].clamp(0.0, 1.0)])
                .collect(),
        );
    }

    Ok(people)
}

pub fn find_spinepose_json_for_image(image_path: &Path) -> Option<PathBuf> {
    let parent = image_path.parent()?;
    let stem = image_path.file_stem()?;
    let mut candidates = Vec::new();

    if let Some(grandparent) = parent.parent() {
        candidates.push(grandparent.join("poses").join(format_path_stem(stem)));
        if let Some(split) = parent.file_name() {
            candidates.push(
                grandparent
                    .join("poses")
                    .join(split)
                    .join(format_path_stem(stem)),
            );
        }
    }

    candidates.push(parent.join("poses").join(format_path_stem(stem)));
    candidates.into_iter().find(|candidate| candidate.is_file())
}

pub fn spinepose_keypoints_to_features(keypoints: &[[f32; 3]]) -> Result<Vec<f32>, RetrievalError> {
    if keypoints.len() < SPINEPOSE_KEYPOINTS {
        return Err(RetrievalError::InvalidData(format!(
            "SpinePose returned {} keypoints, expected {SPINEPOSE_KEYPOINTS}",
            keypoints.len()
        )));
    }

    if !keypoints
        .iter()
        .take(SPINEPOSE_KEYPOINTS)
        .any(|[x, y, confidence]| {
            *confidence >= MIN_KEYPOINT_CONFIDENCE && x.is_finite() && y.is_finite()
        })
    {
        return Ok(vec![0.0; SPINEPOSE_FEATURE_DIM]);
    }

    let (center_x, center_y, scale) = landmark_normalization(keypoints);
    let mut features = Vec::with_capacity(SPINEPOSE_FEATURE_DIM);
    for [x, y, confidence] in keypoints.iter().take(SPINEPOSE_KEYPOINTS).copied() {
        features.push(((x - center_x) / scale).clamp(-1.5, 1.5));
        features.push(((y - center_y) / scale).clamp(-1.5, 1.5));
        features.push(confidence.clamp(0.0, 1.0));
    }
    debug_assert_eq!(features.len(), SPINEPOSE_FEATURE_DIM);
    Ok(features)
}

fn best_spinepose_person(people: &[Vec<[f32; 3]>]) -> Option<&[[f32; 3]]> {
    people
        .iter()
        .filter(|person| person.len() >= SPINEPOSE_KEYPOINTS)
        .max_by(|left, right| {
            person_confidence(left)
                .total_cmp(&person_confidence(right))
                .then_with(|| left.len().cmp(&right.len()))
        })
        .map(Vec::as_slice)
}

fn person_confidence(keypoints: &[[f32; 3]]) -> f32 {
    keypoints
        .iter()
        .take(SPINEPOSE_KEYPOINTS)
        .map(|keypoint| keypoint[2].clamp(0.0, 1.0))
        .sum()
}

fn landmark_normalization(keypoints: &[[f32; 3]]) -> (f32, f32, f32) {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for [x, y, confidence] in keypoints.iter().take(SPINEPOSE_KEYPOINTS) {
        if *confidence < MIN_KEYPOINT_CONFIDENCE || !x.is_finite() || !y.is_finite() {
            continue;
        }
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }

    let width = (max_x - min_x).abs().max(1.0);
    let height = (max_y - min_y).abs().max(1.0);
    (
        min_x + width * 0.5,
        min_y + height * 0.5,
        width.max(height).max(1.0),
    )
}

#[derive(Clone, Copy)]
enum SpinePoseInput<'a> {
    Path(&'a Path),
    EncodedBytes(&'a [u8]),
}

fn run_spinepose(input: SpinePoseInput<'_>) -> Result<Vec<[f32; 3]>, RetrievalError> {
    let python = resolve_spinepose_python()?;
    let mut command = Command::new(&python);
    command.arg("-c").arg(RUNTIME_SCRIPT);
    let (source, stdin_bytes) = match input {
        SpinePoseInput::Path(path) => {
            command.arg(path).stdin(Stdio::null());
            (path.display().to_string(), None)
        }
        SpinePoseInput::EncodedBytes(bytes) => {
            command.arg("-").stdin(Stdio::piped());
            ("encoded image bytes".to_string(), Some(bytes))
        }
    };
    command.stdout(Stdio::piped()).stderr(Stdio::piped());
    if let Ok(home) = env::var("SPINEPOSE_HOME") {
        command.env("HOME", home);
    }

    let mut child = command.spawn().map_err(|error| {
        RetrievalError::InvalidData(format!(
            "failed to start SpinePose runtime with {}: {error}",
            python.display()
        ))
    })?;
    if let Some(bytes) = stdin_bytes {
        let mut stdin = child.stdin.take().ok_or_else(|| {
            RetrievalError::InvalidData("failed to open SpinePose stdin".to_string())
        })?;
        stdin.write_all(bytes)?;
    }

    let output = child.wait_with_output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(RetrievalError::InvalidData(format!(
            "SpinePose failed for {}: {}",
            source,
            stderr.trim()
        )));
    }

    let mut stdout = output.stdout;
    let root: OpenPoseRoot = json::from_slice(&mut stdout)?;
    let people = root
        .people
        .into_iter()
        .map(|person| {
            person
                .pose_keypoints_2d
                .chunks_exact(SPINEPOSE_VALUES_PER_KEYPOINT)
                .take(SPINEPOSE_KEYPOINTS)
                .map(|keypoint| [keypoint[0], keypoint[1], keypoint[2].clamp(0.0, 1.0)])
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    best_spinepose_person(&people)
        .map(<[_]>::to_vec)
        .ok_or_else(|| RetrievalError::InvalidData("SpinePose did not detect a person".to_string()))
}

fn resolve_spinepose_python() -> Result<PathBuf, RetrievalError> {
    if let Ok(path) = env::var("SPINEPOSE_PYTHON") {
        let path = PathBuf::from(path);
        if path.is_file() {
            return Ok(path);
        }
    }

    if let Some(spinepose) = find_in_path("spinepose") {
        let target = fs::read_link(&spinepose).unwrap_or_else(|_| spinepose.clone());
        let target = if target.is_absolute() {
            target
        } else {
            spinepose
                .parent()
                .map(|parent| parent.join(&target))
                .unwrap_or(target)
        };
        if let Some(parent) = target.parent() {
            let python = parent.join("python");
            if python.is_file() {
                return Ok(python);
            }
        }
    }

    for name in ["python3", "python"] {
        if let Some(path) = find_in_path(name) {
            return Ok(path);
        }
    }

    Err(RetrievalError::InvalidData(
        "SpinePose runtime not found; run through `mise run ...` or set SPINEPOSE_PYTHON"
            .to_string(),
    ))
}

fn find_in_path(name: &str) -> Option<PathBuf> {
    env::var_os("PATH").and_then(|paths| {
        env::split_paths(&paths)
            .map(|path| path.join(name))
            .find(|path| path.is_file())
    })
}

fn format_path_stem(stem: &std::ffi::OsStr) -> PathBuf {
    let mut path = PathBuf::new();
    path.push(stem);
    path.set_extension("json");
    path
}

#[derive(Debug, Deserialize)]
struct OpenPoseRoot {
    #[serde(default)]
    people: Vec<OpenPosePerson>,
}

#[derive(Debug, Deserialize)]
struct OpenPosePerson {
    #[serde(default)]
    pose_keypoints_2d: Vec<f32>,
}
