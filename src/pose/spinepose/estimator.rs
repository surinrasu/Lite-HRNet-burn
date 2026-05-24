use std::{
    fs,
    path::{Path, PathBuf},
};

use image::DynamicImage;
use serde::Deserialize;

use super::burn;
use crate::RetrievalError;

pub const SPINEPOSE_KEYPOINTS: usize = 37;
pub const SPINEPOSE_VALUES_PER_KEYPOINT: usize = 3;
pub const SPINEPOSE_FEATURE_DIM: usize = SPINEPOSE_KEYPOINTS * SPINEPOSE_VALUES_PER_KEYPOINT;

const MIN_KEYPOINT_CONFIDENCE: f32 = 0.05;

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

        let image = image::open(path)?;
        let keypoints = run_spinepose(&image)?;
        spinepose_keypoints_to_features(&keypoints)
    }

    pub fn estimate_pose_features_from_bytes(
        &self,
        bytes: &[u8],
    ) -> Result<Vec<f32>, RetrievalError> {
        let image = image::load_from_memory(bytes)?;
        let keypoints = run_spinepose(&image)?;
        spinepose_keypoints_to_features(&keypoints)
    }
}

impl PoseFeatureEstimator for SpinePoseEstimator {
    fn estimate_pose_features(&self, image: &DynamicImage) -> Result<Vec<f32>, RetrievalError> {
        let keypoints = run_spinepose(image)?;
        spinepose_keypoints_to_features(&keypoints)
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

fn run_spinepose(image: &DynamicImage) -> Result<Vec<[f32; 3]>, RetrievalError> {
    let people = burn::estimate_people(image)?;
    best_spinepose_person(&people)
        .map(<[_]>::to_vec)
        .ok_or_else(|| RetrievalError::InvalidData("SpinePose did not detect a person".to_string()))
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
