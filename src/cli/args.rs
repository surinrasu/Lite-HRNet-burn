use std::{fmt, path::PathBuf};

use cli::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use pose_obc_retrieval::{HeadUpsampleMode, LiteHrNetPoseConfig};

extern crate cli as clap;

mod parse;
mod retrieval;

#[cfg(test)]
mod tests;

pub(super) use retrieval::{
    RetrievalIndexArgs, RetrievalSearchArgs, RetrievalServeArgs, RetrievalTrainArgs,
};

use parse::{parse_input_size, parse_positive_count, parse_positive_f32, parse_positive_f64};

#[derive(Debug, Parser)]
#[command(name = "pose-obc-retrieval", version, arg_required_else_help = true)]
pub(super) struct Cli {
    #[command(subcommand)]
    pub(super) command: Command,
}

#[derive(Debug, Subcommand)]
pub(super) enum Command {
    /// Train a model.
    Train(TrainArgs),
    /// Precompute candidate glyph embeddings into a JSON index.
    Index(RetrievalIndexArgs),
    /// Run a top-k retrieval query from an image or dataset sample.
    Search(RetrievalSearchArgs),
    /// Serve the browser UI for upload/sample queries.
    Serve(RetrievalServeArgs),
}

#[derive(Debug, Args)]
pub(super) struct TrainArgs {
    #[command(subcommand)]
    pub(super) target: TrainTarget,
}

#[derive(Debug, Subcommand)]
pub(super) enum TrainTarget {
    /// Train the pose/glyph retrieval model.
    Retrieval(RetrievalTrainArgs),
    /// Train on COCO person-keypoints annotations.
    Pose(Box<PoseTrainArgs>),
}

#[derive(Clone, Debug, Args)]
pub(super) struct PoseTrainArgs {
    /// COCO person-keypoints annotation JSON.
    #[arg(long = "annotations", value_name = "PATH")]
    pub(super) train_ann: PathBuf,
    /// Directory containing the training images referenced by the annotations.
    #[arg(long = "images", value_name = "DIR")]
    pub(super) train_images: PathBuf,
    /// Directory containing SpinePose JSON files for the training images.
    #[arg(long = "pose-dir", value_name = "DIR")]
    pub(super) train_pose_dir: Option<PathBuf>,
    /// Validation COCO person-keypoints annotation JSON.
    #[arg(long = "validation-annotations", value_name = "PATH")]
    pub(super) val_ann: Option<PathBuf>,
    /// Validation image directory. Defaults to --images when validation annotations are provided.
    #[arg(long = "validation-images", value_name = "DIR", requires = "val_ann")]
    pub(super) val_images: Option<PathBuf>,
    /// Directory containing SpinePose JSON files for the validation images.
    #[arg(long = "validation-pose-dir", value_name = "DIR", requires = "val_ann")]
    pub(super) val_pose_dir: Option<PathBuf>,
    /// Directory for checkpoints and the training report.
    #[arg(
        short = 'o',
        long = "output-dir",
        value_name = "DIR",
        default_value = "runs/litehrnet"
    )]
    pub(super) out_dir: PathBuf,
    /// Burn backend to use.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    pub(super) backend: BackendArg,
    /// Lite-HRNet model variant.
    #[arg(long, value_enum, default_value_t = ModelArg::LiteHrNet18)]
    pub(super) model: ModelArg,
    /// Training epochs.
    #[arg(short = 'e', long, default_value_t = 210, value_parser = parse_positive_count)]
    pub(super) epochs: usize,
    /// Batch size.
    #[arg(short = 'b', long, default_value_t = 32, value_parser = parse_positive_count)]
    pub(super) batch_size: usize,
    /// Adam learning rate.
    #[arg(long = "learning-rate", value_name = "LR", default_value_t = 2e-3, value_parser = parse_positive_f64)]
    pub(super) learning_rate: f64,
    /// Input tensor size as HEIGHTxWIDTH.
    #[arg(long = "input-size", value_name = "HEIGHTxWIDTH", default_value = "256x192", value_parser = parse_input_size)]
    pub(super) input_size: InputSize,
    /// Heatmap Gaussian sigma.
    #[arg(long, default_value_t = 2.0, value_parser = parse_positive_f32)]
    pub(super) sigma: f32,
    /// Limit the number of training samples used from the dataset.
    #[arg(long = "max-samples", value_name = "N", value_parser = parse_positive_count)]
    pub(super) max_train_samples: Option<usize>,
    /// Limit the number of validation samples used from the dataset.
    #[arg(
        long = "max-validation-samples",
        value_name = "N",
        value_parser = parse_positive_count,
        requires = "val_ann"
    )]
    pub(super) max_val_samples: Option<usize>,
    /// Print batch progress every N batches. Use 0 for epoch-only progress.
    #[arg(long, default_value_t = 50)]
    pub(super) log_every: usize,
    /// CPU batches to prepare ahead of the GPU step. Defaults to 2 on metal and 0 on flex.
    #[arg(long = "prefetch-batches")]
    pub(super) prefetch_batches: Option<usize>,
    /// RNG seed used for sample shuffling.
    #[arg(long, default_value_t = 42)]
    pub(super) seed: u64,
    /// Disable dataset shuffling.
    #[arg(long = "no-shuffle", action = ArgAction::SetFalse, default_value_t = true)]
    pub(super) shuffle: bool,
    /// Disable per-epoch checkpoints.
    #[arg(long = "no-save-every-epoch", action = ArgAction::SetFalse, default_value_t = true)]
    pub(super) save_every_epoch: bool,
    /// Upsampling mode used by the pose head. Defaults to bilinear on flex and nearest on metal.
    #[arg(long = "head-upsample", value_enum)]
    pub(super) head_upsample_mode: Option<HeadUpsampleArg>,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct InputSize {
    pub(super) height: usize,
    pub(super) width: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(super) enum BackendArg {
    Flex,
    Metal,
}

impl fmt::Display for BackendArg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Flex => "flex",
            Self::Metal => "metal",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(super) enum ModelArg {
    #[value(name = "litehrnet18")]
    LiteHrNet18,
    #[value(name = "litehrnet30")]
    LiteHrNet30,
}

impl ModelArg {
    pub(super) fn config(self) -> LiteHrNetPoseConfig {
        match self {
            Self::LiteHrNet18 => LiteHrNetPoseConfig::litehrnet18_coco(),
            Self::LiteHrNet30 => LiteHrNetPoseConfig::litehrnet30_coco(),
        }
    }
}

impl fmt::Display for ModelArg {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::LiteHrNet18 => "litehrnet18",
            Self::LiteHrNet30 => "litehrnet30",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(super) enum HeadUpsampleArg {
    Bilinear,
    Nearest,
}

impl HeadUpsampleArg {
    pub(super) fn mode(self) -> HeadUpsampleMode {
        match self {
            Self::Bilinear => HeadUpsampleMode::BilinearAligned,
            Self::Nearest => HeadUpsampleMode::Nearest,
        }
    }
}
