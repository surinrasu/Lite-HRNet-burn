use std::{fs, path::Path, time::UNIX_EPOCH};

use serde::{Deserialize, Serialize};

use super::{
    RETRIEVAL_FEATURE_DIM, RetrievalError, canonical_or_original, ensure_feature_dim,
    ensure_finite_values, extract_glyph_features_from_path, read_json_file, write_json_file,
};
use crate::pose::spinepose::{DefaultPoseEstimator, find_spinepose_json_for_image};

const FEATURE_CACHE_VERSION: u32 = 1;

pub(crate) fn extract_glyph_features_with_cache(
    path: &Path,
    feature_cache_dir: Option<&Path>,
) -> Result<Vec<f32>, RetrievalError> {
    cached_or_extract_features(feature_cache_dir, FeatureKind::Glyph, path, None, || {
        extract_glyph_features_from_path(path)
    })
}

pub(crate) fn extract_pose_features_with_cache(
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
enum FeatureKind {
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
