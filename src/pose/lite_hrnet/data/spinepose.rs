use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use super::PoseDataError;
use crate::pose::spinepose::{SPINEPOSE_KEYPOINTS, read_spinepose_people};

type SpinePosePerson = Vec<[f32; 3]>;
type SpinePosePeople = Vec<SpinePosePerson>;
pub(super) type SpinePosePeopleByImage = HashMap<u64, SpinePosePeople>;

pub(super) fn load_spinepose_people_by_image<'a>(
    spinepose_root: &Path,
    images: impl IntoIterator<Item = (u64, &'a str)>,
) -> Result<SpinePosePeopleByImage, PoseDataError> {
    let mut output = HashMap::new();
    for (image_id, file_name) in images {
        let path = spinepose_path_for_image(spinepose_root, file_name);
        if !path.is_file() {
            continue;
        }
        let people = read_spinepose_people(&path).map_err(|error| {
            PoseDataError::InvalidDataset(format!(
                "invalid SpinePose JSON {}: {error}",
                path.display()
            ))
        })?;
        output.insert(image_id, people);
    }
    Ok(output)
}

fn spinepose_path_for_image(spinepose_root: &Path, image_file_name: &str) -> PathBuf {
    let mut relative = PathBuf::from(image_file_name);
    relative.set_extension("json");
    spinepose_root.join(relative)
}

pub(super) fn match_spinepose_person(
    people: &[SpinePosePerson],
    bbox: [f32; 4],
    num_joints: usize,
) -> Option<Vec<[f32; 3]>> {
    if num_joints > SPINEPOSE_KEYPOINTS {
        return None;
    }

    let annotation_center = [bbox[0] + bbox[2] * 0.5, bbox[1] + bbox[3] * 0.5];
    let annotation_diag = (bbox[2].hypot(bbox[3])).max(1.0);
    let mut best = None;
    let mut best_score = f32::NEG_INFINITY;

    for person in people {
        if person.len() < num_joints {
            continue;
        }
        let Some(person_bbox) = keypoint_bbox(person) else {
            continue;
        };
        let person_center = [
            person_bbox[0] + person_bbox[2] * 0.5,
            person_bbox[1] + person_bbox[3] * 0.5,
        ];
        let center_distance = (annotation_center[0] - person_center[0])
            .hypot(annotation_center[1] - person_center[1])
            / annotation_diag;
        let score =
            bbox_iou(bbox, person_bbox) * 2.0 - center_distance + average_confidence(person) * 0.01;

        if score > best_score {
            best_score = score;
            best = Some(person.iter().take(num_joints).copied().collect::<Vec<_>>());
        }
    }

    best
}

fn keypoint_bbox(keypoints: &[[f32; 3]]) -> Option<[f32; 4]> {
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for [x, y, confidence] in keypoints {
        if *confidence <= 0.05 || !x.is_finite() || !y.is_finite() {
            continue;
        }
        min_x = min_x.min(*x);
        min_y = min_y.min(*y);
        max_x = max_x.max(*x);
        max_y = max_y.max(*y);
    }

    min_x.is_finite().then_some([
        min_x,
        min_y,
        (max_x - min_x).max(1.0),
        (max_y - min_y).max(1.0),
    ])
}

fn bbox_iou(left: [f32; 4], right: [f32; 4]) -> f32 {
    let left_x2 = left[0] + left[2];
    let left_y2 = left[1] + left[3];
    let right_x2 = right[0] + right[2];
    let right_y2 = right[1] + right[3];

    let intersection_width = (left_x2.min(right_x2) - left[0].max(right[0])).max(0.0);
    let intersection_height = (left_y2.min(right_y2) - left[1].max(right[1])).max(0.0);
    let intersection = intersection_width * intersection_height;
    let union = left[2] * left[3] + right[2] * right[3] - intersection;
    if union <= 0.0 {
        0.0
    } else {
        intersection / union
    }
}

fn average_confidence(keypoints: &[[f32; 3]]) -> f32 {
    let mut total = 0.0;
    let mut count = 0.0;
    for keypoint in keypoints.iter().take(SPINEPOSE_KEYPOINTS) {
        total += keypoint[2].clamp(0.0, 1.0);
        count += 1.0;
    }
    if count > 0.0 { total / count } else { 0.0 }
}
