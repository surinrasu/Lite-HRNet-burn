extern crate ann as burn;

pub mod pose;
pub mod retrieval;
pub mod service;

pub use pose::lite_hrnet::{
    ActivationKind, BatchTrainingProgress, CocoPoseDataset, ConditionalChannelWeighting, ConvBnAct,
    CrossResolutionWeighting, DEFAULT_POSE_JOINTS, DepthwiseSeparableConv, EpochReport, FuseLayer,
    HeadUpsampleMode, IterativeHead, LiteHrModule, LiteHrModuleType, LiteHrNet, LiteHrNetConfig,
    LiteHrNetPose, LiteHrNetPoseConfig, PoseBatch, PoseDataConfig, PoseDataError, PoseSample,
    PoseTensorBatch, PoseTensorSample, PoseTrainError, PoseTrainingConfig, PoseTrainingProgress,
    PoseTrainingReport, ShuffleUnit, SpatialWeighting, Stage, Stem, TransitionBranch,
    TransitionLayer, channel_shuffle, evaluate_dataset, interpolate_bilinear_aligned,
    interpolate_nearest, joints_mse_loss, run_synthetic_training, synthetic_pose_batch,
    train_dataset, train_dataset_with_progress, train_step, train_step_with_loss,
    train_step_with_loss_tensor,
};
pub use pose::spinepose::{
    DefaultPoseEstimator, PoseFeatureEstimator, SPINEPOSE_FEATURE_DIM, SPINEPOSE_KEYPOINTS,
    SPINEPOSE_VALUES_PER_KEYPOINT, SpinePoseEstimator, estimate_pose_features_from_bytes,
    estimate_pose_features_from_image, estimate_pose_features_from_path,
    find_spinepose_json_for_image, read_spinepose_features, read_spinepose_people,
    spinepose_keypoints_to_features,
};
pub use retrieval::*;
pub use service::{RetrievalService, serve_retrieval};
