use std::{fmt, path::PathBuf};

use cli::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use pose_obc_retrieval::{HeadUpsampleMode, LiteHrNetPoseConfig};

extern crate cli as clap;

#[derive(Debug, Parser)]
#[command(name = "pose-obc-retrieval", version, arg_required_else_help = true)]
pub(super) struct Cli {
    #[command(subcommand)]
    pub(super) command: Command,
}

#[derive(Debug, Subcommand)]
pub(super) enum Command {
    /// Train on COCO person-keypoints annotations.
    Train(TrainArgs),
    /// Run a synthetic forward/backward optimizer smoke check.
    Smoke(SmokeArgs),
    /// Train, index, search, and serve the oracle-bone pose retrieval system.
    Retrieval(RetrievalArgs),
}

#[derive(Debug, Args)]
pub(super) struct RetrievalArgs {
    #[command(subcommand)]
    pub(super) command: RetrievalCommand,
}

#[derive(Debug, Subcommand)]
pub(super) enum RetrievalCommand {
    /// Train the pose/glyph twin-tower retrieval model from data/persona_* pairs.
    Train(RetrievalTrainArgs),
    /// Precompute candidate glyph embeddings into a JSON index.
    Index(RetrievalIndexArgs),
    /// Run a single top-k retrieval query from an image or dataset sample.
    Search(RetrievalSearchArgs),
    /// Serve the browser UI for upload/sample queries.
    Serve(RetrievalServeArgs),
}

#[derive(Clone, Debug, Args)]
pub(super) struct TrainArgs {
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

#[derive(Clone, Debug, Args)]
pub(super) struct SmokeArgs {
    /// Burn backend to use.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    pub(super) backend: BackendArg,
    /// Lite-HRNet model variant.
    #[arg(long, value_enum, default_value_t = ModelArg::LiteHrNet18)]
    pub(super) model: ModelArg,
    /// Optimizer steps to run.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_count)]
    pub(super) steps: usize,
    /// Batch size.
    #[arg(short = 'b', long, default_value_t = 1, value_parser = parse_positive_count)]
    pub(super) batch_size: usize,
    /// Adam learning rate.
    #[arg(long = "learning-rate", value_name = "LR", default_value_t = 2e-3, value_parser = parse_positive_f64)]
    pub(super) learning_rate: f64,
    /// Input tensor size as HEIGHTxWIDTH.
    #[arg(long = "input-size", value_name = "HEIGHTxWIDTH", default_value = "64x48", value_parser = parse_input_size)]
    pub(super) input_size: InputSize,
    /// Upsampling mode used by the pose head. Defaults to bilinear on flex and nearest on metal.
    #[arg(long = "head-upsample", value_enum)]
    pub(super) head_upsample_mode: Option<HeadUpsampleArg>,
}

#[derive(Clone, Debug, Args)]
pub(super) struct RetrievalTrainArgs {
    /// Burn backend to use.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    pub(super) backend: BackendArg,
    /// Root containing data/persona_*/images and data/persona_*/glyphs.
    #[arg(long = "data-root", value_name = "DIR", default_value = "data")]
    pub(super) data_root: PathBuf,
    /// Directory for retrieval checkpoints, config, and training report.
    #[arg(
        short = 'o',
        long = "output-dir",
        value_name = "DIR",
        default_value = "runs/retrieval"
    )]
    pub(super) out_dir: PathBuf,
    /// Training epochs.
    #[arg(short = 'e', long, default_value_t = 20, value_parser = parse_positive_count)]
    pub(super) epochs: usize,
    /// Batch size.
    #[arg(short = 'b', long, default_value_t = 32, value_parser = parse_positive_count)]
    pub(super) batch_size: usize,
    /// Adam learning rate.
    #[arg(long = "learning-rate", value_name = "LR", default_value_t = 1e-3, value_parser = parse_positive_f64)]
    pub(super) learning_rate: f64,
    /// Hidden dimension of each MLP tower.
    #[arg(long = "hidden-dim", default_value_t = 128, value_parser = parse_positive_count)]
    pub(super) hidden_dim: usize,
    /// Shared embedding dimension used for cosine search.
    #[arg(long = "embedding-dim", default_value_t = 64, value_parser = parse_positive_count)]
    pub(super) embedding_dim: usize,
    /// Contrastive softmax temperature.
    #[arg(long, default_value_t = 0.07, value_parser = parse_positive_f64)]
    pub(super) temperature: f64,
    /// Limit the number of paired training samples.
    #[arg(long = "max-pairs", value_name = "N", value_parser = parse_positive_count)]
    pub(super) max_pairs: Option<usize>,
    /// Print batch progress every N batches. Use 0 for epoch-only progress.
    #[arg(long, default_value_t = 20)]
    pub(super) log_every: usize,
    /// RNG seed used for sample shuffling.
    #[arg(long, default_value_t = 42)]
    pub(super) seed: u64,
    /// Disable dataset shuffling.
    #[arg(long = "no-shuffle", action = ArgAction::SetFalse, default_value_t = true)]
    pub(super) shuffle: bool,
    /// Save epoch_NNN.mpk checkpoints in addition to last.mpk.
    #[arg(long = "save-every-epoch", action = ArgAction::SetTrue)]
    pub(super) save_every_epoch: bool,
}

#[derive(Clone, Debug, Args)]
pub(super) struct RetrievalIndexArgs {
    /// Burn backend to use for candidate encoding.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    pub(super) backend: BackendArg,
    /// Root containing data/persona_*/images and data/persona_*/glyphs.
    #[arg(long = "data-root", value_name = "DIR", default_value = "data")]
    pub(super) data_root: PathBuf,
    /// Retrieval model checkpoint, with or without the .mpk extension.
    #[arg(long, value_name = "PATH", default_value = "runs/retrieval/last.mpk")]
    pub(super) model: PathBuf,
    /// Retrieval model config JSON. Defaults to retrieval_config.json next to --model.
    #[arg(long, value_name = "PATH")]
    pub(super) config: Option<PathBuf>,
    /// Output candidate embedding index.
    #[arg(
        short = 'o',
        long,
        value_name = "PATH",
        default_value = "runs/retrieval/glyph_index.json"
    )]
    pub(super) output: PathBuf,
    /// Keep duplicate glyph candidates from different persona directories.
    #[arg(long = "include-duplicate-glyphs", action = ArgAction::SetFalse, default_value_t = true)]
    pub(super) unique_glyphs: bool,
}

#[derive(Clone, Debug, Args)]
pub(super) struct RetrievalSearchArgs {
    /// Burn backend to use for query encoding.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    pub(super) backend: BackendArg,
    /// Candidate embedding index JSON.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "runs/retrieval/glyph_index.json"
    )]
    pub(super) index: PathBuf,
    /// Retrieval model checkpoint, with or without the .mpk extension.
    #[arg(long, value_name = "PATH", default_value = "runs/retrieval/last.mpk")]
    pub(super) model: PathBuf,
    /// Retrieval model config JSON. Defaults to config stored in --index or next to --model.
    #[arg(long, value_name = "PATH")]
    pub(super) config: Option<PathBuf>,
    /// Root containing data/persona_*/images and data/persona_*/glyphs, used with --sample.
    #[arg(long = "data-root", value_name = "DIR", default_value = "data")]
    pub(super) data_root: PathBuf,
    /// Query image path.
    #[arg(long, value_name = "PATH", conflicts_with = "sample")]
    pub(super) image: Option<PathBuf>,
    /// Query by pair index from the data directory.
    #[arg(long, value_name = "N", conflicts_with = "image")]
    pub(super) sample: Option<usize>,
    /// Number of hits to return.
    #[arg(short = 'k', long = "top-k", default_value_t = 8, value_parser = parse_positive_count)]
    pub(super) top_k: usize,
}

#[derive(Clone, Debug, Args)]
pub(super) struct RetrievalServeArgs {
    /// Burn backend to use for query and candidate encoding.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    pub(super) backend: BackendArg,
    /// Address for the HTTP UI.
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub(super) addr: String,
    /// Root containing data/persona_*/images and data/persona_*/glyphs.
    #[arg(long = "data-root", value_name = "DIR", default_value = "data")]
    pub(super) data_root: PathBuf,
    /// Optional precomputed candidate embedding index JSON.
    #[arg(long, value_name = "PATH")]
    pub(super) index: Option<PathBuf>,
    /// Retrieval model checkpoint, with or without the .mpk extension.
    #[arg(long, value_name = "PATH", default_value = "runs/retrieval/last.mpk")]
    pub(super) model: PathBuf,
    /// Retrieval model config JSON. Defaults to config stored in --index or next to --model.
    #[arg(long, value_name = "PATH")]
    pub(super) config: Option<PathBuf>,
    /// Default number of hits in the UI.
    #[arg(short = 'k', long = "top-k", default_value_t = 8, value_parser = parse_positive_count)]
    pub(super) top_k: usize,
    /// Enable live browser video frame retrieval/scoring.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(super) live: bool,
    /// Keep duplicate glyph candidates from different persona directories when building in memory.
    #[arg(long = "include-duplicate-glyphs", action = ArgAction::SetFalse, default_value_t = true)]
    pub(super) unique_glyphs: bool,
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

fn parse_input_size(value: &str) -> Result<InputSize, String> {
    let (height, width) = value
        .split_once('x')
        .or_else(|| value.split_once('X'))
        .ok_or_else(|| "expected HEIGHTxWIDTH, for example 256x192".to_string())?;
    let height = parse_positive_usize(height, "height")?;
    let width = parse_positive_usize(width, "width")?;

    Ok(InputSize { height, width })
}

fn parse_positive_usize(value: &str, label: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {label} `{value}`: {error}"))?;
    if parsed == 0 {
        Err(format!("{label} must be greater than 0"))
    } else {
        Ok(parsed)
    }
}

fn parse_positive_count(value: &str) -> Result<usize, String> {
    parse_positive_usize(value, "value")
}

fn parse_positive_f64(value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("invalid float `{value}`: {error}"))?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err("value must be a finite number greater than 0".to_string())
    }
}

fn parse_positive_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|error| format!("invalid float `{value}`: {error}"))?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err("value must be a finite number greater than 0".to_string())
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_train_command() {
        let cli = Cli::parse_from([
            "pose-obc-retrieval",
            "train",
            "--annotations",
            "person_keypoints_train.json",
            "--images",
            "train2017",
            "--pose-dir",
            "poses/train2017",
            "--input-size",
            "128x96",
            "--model",
            "litehrnet30",
        ]);

        let Command::Train(args) = cli.command else {
            panic!("expected train command");
        };
        assert_eq!(args.train_ann, PathBuf::from("person_keypoints_train.json"));
        assert_eq!(args.train_images, PathBuf::from("train2017"));
        assert_eq!(args.train_pose_dir, Some(PathBuf::from("poses/train2017")));
        assert_eq!(args.input_size.height, 128);
        assert_eq!(args.input_size.width, 96);
        assert_eq!(args.model, ModelArg::LiteHrNet30);
    }

    #[test]
    fn parses_smoke_command() {
        let cli = Cli::parse_from([
            "pose-obc-retrieval",
            "smoke",
            "--backend",
            "metal",
            "--steps",
            "2",
            "--head-upsample",
            "nearest",
        ]);

        let Command::Smoke(args) = cli.command else {
            panic!("expected smoke command");
        };
        assert_eq!(args.backend, BackendArg::Metal);
        assert_eq!(args.steps, 2);
        assert_eq!(args.head_upsample_mode, Some(HeadUpsampleArg::Nearest));
    }

    #[test]
    fn parses_retrieval_backend_argument() {
        let cli = Cli::parse_from([
            "pose-obc-retrieval",
            "retrieval",
            "search",
            "--backend",
            "metal",
            "--sample",
            "1",
        ]);

        let Command::Retrieval(args) = cli.command else {
            panic!("expected retrieval command");
        };
        let RetrievalCommand::Search(args) = args.command else {
            panic!("expected retrieval search command");
        };
        assert_eq!(args.backend, BackendArg::Metal);
        assert_eq!(args.sample, Some(1));
    }

    #[test]
    fn parses_retrieval_serve_live_argument() {
        let cli = Cli::parse_from([
            "pose-obc-retrieval",
            "retrieval",
            "serve",
            "--live",
            "--top-k",
            "5",
        ]);

        let Command::Retrieval(args) = cli.command else {
            panic!("expected retrieval command");
        };
        let RetrievalCommand::Serve(args) = args.command else {
            panic!("expected retrieval serve command");
        };
        assert!(args.live);
        assert_eq!(args.top_k, 5);
    }

    #[test]
    fn rejects_invalid_input_size() {
        let error = Cli::try_parse_from(["pose-obc-retrieval", "smoke", "--input-size", "64"])
            .expect_err("invalid input size should fail");

        assert!(error.to_string().contains("HEIGHTxWIDTH"));
    }

    #[test]
    fn rejects_validation_images_without_validation_annotations() {
        let error = Cli::try_parse_from([
            "pose-obc-retrieval",
            "train",
            "--annotations",
            "person_keypoints_train.json",
            "--images",
            "train2017",
            "--validation-images",
            "val2017",
        ])
        .expect_err("validation images without annotations should fail");

        assert!(error.to_string().contains("--validation-annotations"));
    }
}
