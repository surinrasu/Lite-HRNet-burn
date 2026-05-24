mod cache;
mod dataset;
mod error;
mod features;
mod index;
mod io;
mod model;
mod train;

pub use dataset::*;
pub use error::*;
pub use features::{
    extract_glyph_features_from_path, extract_pose_features_from_bytes,
    extract_pose_features_from_path,
};
pub use index::*;
pub use model::*;
pub use train::*;

pub(crate) use cache::{extract_glyph_features_with_cache, extract_pose_features_with_cache};
pub(crate) use features::{ensure_feature_dim, ensure_finite_values};
pub(crate) use io::{canonical_or_original, read_json_file, recorder_path_stem, write_json_file};

use crate::pose::spinepose::{SPINEPOSE_FEATURE_DIM, SPINEPOSE_KEYPOINTS};

pub const RETRIEVAL_KEYPOINTS: usize = SPINEPOSE_KEYPOINTS;
pub const RETRIEVAL_FEATURE_DIM: usize = SPINEPOSE_FEATURE_DIM;
pub const DEFAULT_RETRIEVAL_HIDDEN_DIM: usize = 128;
pub const DEFAULT_RETRIEVAL_EMBEDDING_DIM: usize = 64;
pub const CANDIDATE_INDEX_VERSION: u32 = 1;
