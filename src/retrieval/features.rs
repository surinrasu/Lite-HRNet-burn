use std::path::Path;

use image::{DynamicImage, GenericImageView, ImageReader};

use super::{RETRIEVAL_FEATURE_DIM, RETRIEVAL_KEYPOINTS, RetrievalError};
use crate::pose::spinepose::{estimate_pose_features_from_bytes, estimate_pose_features_from_path};

const FEATURE_POINTS_PER_BAND: usize = 3;
const FEATURE_BANDS: usize = RETRIEVAL_KEYPOINTS.div_ceil(FEATURE_POINTS_PER_BAND);

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

pub(crate) fn ensure_feature_dim(actual: usize, expected: usize) -> Result<(), RetrievalError> {
    if actual == expected {
        Ok(())
    } else {
        Err(RetrievalError::InvalidData(format!(
            "feature dimension mismatch: expected {expected}, got {actual}"
        )))
    }
}

pub(crate) fn ensure_finite_values(label: &str, values: &[f32]) -> Result<(), RetrievalError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        Err(RetrievalError::InvalidData(format!(
            "{label} at index {index} is not finite"
        )))
    } else {
        Ok(())
    }
}
