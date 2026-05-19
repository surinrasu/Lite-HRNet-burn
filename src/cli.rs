use std::{error::Error, fmt, path::PathBuf};

use burn::{backend::Autodiff, tensor::backend::AutodiffBackend};
use clap::{ArgAction, Args, Parser, Subcommand, ValueEnum};
use lite_hrnet_burn::{
    CocoPoseDataset, HeadUpsampleMode, LiteHrNetPoseConfig, PoseDataConfig, PoseTrainingConfig,
    PoseTrainingProgress, PoseTrainingReport,
    train::{run_synthetic_training, train_dataset_with_progress},
};

pub fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Train(args) => run_train(args),
        Command::Smoke(args) => run_smoke(args),
    }
}

#[derive(Debug, Parser)]
#[command(
    name = "lite-hrnet-burn",
    version,
    about = "Train and check Lite-HRNet pose models with Burn",
    arg_required_else_help = true
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Train on COCO person-keypoints annotations.
    Train(TrainArgs),
    /// Run a synthetic forward/backward optimizer smoke check.
    Smoke(SmokeArgs),
}

#[derive(Clone, Debug, Args)]
struct TrainArgs {
    /// COCO person-keypoints annotation JSON.
    #[arg(long = "annotations", value_name = "PATH")]
    train_ann: PathBuf,
    /// Directory containing the training images referenced by the annotations.
    #[arg(long = "images", value_name = "DIR")]
    train_images: PathBuf,
    /// Validation COCO person-keypoints annotation JSON.
    #[arg(long = "validation-annotations", value_name = "PATH")]
    val_ann: Option<PathBuf>,
    /// Validation image directory. Defaults to --images when validation annotations are provided.
    #[arg(long = "validation-images", value_name = "DIR", requires = "val_ann")]
    val_images: Option<PathBuf>,
    /// Directory for checkpoints and the training report.
    #[arg(
        short = 'o',
        long = "output-dir",
        value_name = "DIR",
        default_value = "runs/litehrnet"
    )]
    out_dir: PathBuf,
    /// Burn backend to use.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    backend: BackendArg,
    /// Lite-HRNet model variant.
    #[arg(long, value_enum, default_value_t = ModelArg::LiteHrNet18)]
    model: ModelArg,
    /// Training epochs.
    #[arg(short = 'e', long, default_value_t = 210, value_parser = parse_positive_count)]
    epochs: usize,
    /// Batch size.
    #[arg(short = 'b', long, default_value_t = 32, value_parser = parse_positive_count)]
    batch_size: usize,
    /// Adam learning rate.
    #[arg(long = "learning-rate", value_name = "LR", default_value_t = 2e-3, value_parser = parse_positive_f64)]
    learning_rate: f64,
    /// Input tensor size as HEIGHTxWIDTH.
    #[arg(long = "input-size", value_name = "HEIGHTxWIDTH", default_value = "256x192", value_parser = parse_input_size)]
    input_size: InputSize,
    /// Heatmap Gaussian sigma.
    #[arg(long, default_value_t = 2.0, value_parser = parse_positive_f32)]
    sigma: f32,
    /// Limit the number of training samples used from the dataset.
    #[arg(long = "max-samples", value_name = "N", value_parser = parse_positive_count)]
    max_train_samples: Option<usize>,
    /// Limit the number of validation samples used from the dataset.
    #[arg(
        long = "max-validation-samples",
        value_name = "N",
        value_parser = parse_positive_count,
        requires = "val_ann"
    )]
    max_val_samples: Option<usize>,
    /// Print batch progress every N batches. Use 0 for epoch-only progress.
    #[arg(long, default_value_t = 50)]
    log_every: usize,
    /// RNG seed used for sample shuffling.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Disable dataset shuffling.
    #[arg(long = "no-shuffle", action = ArgAction::SetFalse, default_value_t = true)]
    shuffle: bool,
    /// Disable per-epoch checkpoints.
    #[arg(long = "no-save-every-epoch", action = ArgAction::SetFalse, default_value_t = true)]
    save_every_epoch: bool,
    /// Upsampling mode used by the pose head. Defaults to bilinear on flex and nearest on metal.
    #[arg(long = "head-upsample", value_enum)]
    head_upsample_mode: Option<HeadUpsampleArg>,
}

#[derive(Clone, Debug, Args)]
struct SmokeArgs {
    /// Burn backend to use.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    backend: BackendArg,
    /// Lite-HRNet model variant.
    #[arg(long, value_enum, default_value_t = ModelArg::LiteHrNet18)]
    model: ModelArg,
    /// Optimizer steps to run.
    #[arg(long, default_value_t = 1, value_parser = parse_positive_count)]
    steps: usize,
    /// Batch size.
    #[arg(short = 'b', long, default_value_t = 1, value_parser = parse_positive_count)]
    batch_size: usize,
    /// Adam learning rate.
    #[arg(long = "learning-rate", value_name = "LR", default_value_t = 2e-3, value_parser = parse_positive_f64)]
    learning_rate: f64,
    /// Input tensor size as HEIGHTxWIDTH.
    #[arg(long = "input-size", value_name = "HEIGHTxWIDTH", default_value = "64x48", value_parser = parse_input_size)]
    input_size: InputSize,
    /// Upsampling mode used by the pose head. Defaults to bilinear on flex and nearest on metal.
    #[arg(long = "head-upsample", value_enum)]
    head_upsample_mode: Option<HeadUpsampleArg>,
}

#[derive(Clone, Copy, Debug)]
struct InputSize {
    height: usize,
    width: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BackendArg {
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
enum ModelArg {
    #[value(name = "litehrnet18")]
    LiteHrNet18,
    #[value(name = "litehrnet30")]
    LiteHrNet30,
}

impl ModelArg {
    fn config(self) -> LiteHrNetPoseConfig {
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
enum HeadUpsampleArg {
    Bilinear,
    Nearest,
}

impl HeadUpsampleArg {
    fn mode(self) -> HeadUpsampleMode {
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

fn run_train(args: TrainArgs) -> Result<(), Box<dyn Error>> {
    match args.backend {
        BackendArg::Flex => {
            type Backend = Autodiff<burn::backend::Flex>;
            let device = Default::default();
            train_with_backend::<Backend>(args, &device)
        }
        BackendArg::Metal => train_metal(args),
    }
}

#[cfg(feature = "metal")]
fn train_metal(args: TrainArgs) -> Result<(), Box<dyn Error>> {
    use burn::backend::Metal;

    type Backend = Autodiff<Metal>;
    let device = Default::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
        &device,
        Default::default(),
    );
    train_with_backend::<Backend>(args, &device)
}

#[cfg(not(feature = "metal"))]
fn train_metal(_args: TrainArgs) -> Result<(), Box<dyn Error>> {
    Err("rebuild with `--features metal` to train on Metal".into())
}

fn train_with_backend<B: AutodiffBackend>(
    args: TrainArgs,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    let checkpoint_dir = args.out_dir.clone();
    let total_epochs = args.epochs;
    let data = PoseDataConfig {
        sigma: args.sigma,
        ..PoseDataConfig::from_input(args.input_size.height, args.input_size.width, 17)
    };

    let train_data = CocoPoseDataset::from_coco(&args.train_ann, &args.train_images, data.clone())?;
    let val_dataset = match &args.val_ann {
        Some(val_ann) => Some(CocoPoseDataset::from_coco(
            val_ann,
            args.val_images.as_ref().unwrap_or(&args.train_images),
            data,
        )?),
        None => None,
    };

    let mut model = args.model.config();
    model.backbone.head_upsample_mode =
        resolve_head_upsample(args.backend, args.head_upsample_mode);

    let train_samples = limited_len(train_data.len(), args.max_train_samples);
    let val_samples = val_dataset
        .as_ref()
        .map(|dataset| limited_len(dataset.len(), args.max_val_samples));
    print_train_start(
        &args,
        train_samples,
        val_samples,
        model.backbone.head_upsample_mode,
    );

    let config = PoseTrainingConfig {
        model,
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        shuffle: args.shuffle,
        seed: args.seed,
        max_train_samples: args.max_train_samples,
        max_val_samples: args.max_val_samples,
        checkpoint_dir: args.out_dir,
        log_every: args.log_every,
        save_every_epoch: args.save_every_epoch,
    };

    let (_model, report) =
        train_dataset_with_progress::<B, _>(config, train_data, val_dataset, device, |progress| {
            print_training_progress(progress, total_epochs);
        })?;
    print_training_done(&report, &checkpoint_dir);
    Ok(())
}

fn run_smoke(args: SmokeArgs) -> Result<(), Box<dyn Error>> {
    let backend = args.backend;
    match backend {
        BackendArg::Flex => {
            print_smoke_start(&args);
            type Backend = Autodiff<burn::backend::Flex>;
            let device = Default::default();
            smoke_with_backend::<Backend>(args, &device)?;
            print_smoke_done(backend);
            Ok(())
        }
        BackendArg::Metal => {
            print_smoke_start(&args);
            smoke_metal(args)?;
            print_smoke_done(backend);
            Ok(())
        }
    }
}

#[cfg(feature = "metal")]
fn smoke_metal(args: SmokeArgs) -> Result<(), Box<dyn Error>> {
    use burn::backend::Metal;

    type Backend = Autodiff<Metal>;
    let device = Default::default();
    burn::backend::wgpu::init_setup::<burn::backend::wgpu::graphics::Metal>(
        &device,
        Default::default(),
    );
    smoke_with_backend::<Backend>(args, &device)?;
    Ok(())
}

#[cfg(not(feature = "metal"))]
fn smoke_metal(_args: SmokeArgs) -> Result<(), Box<dyn Error>> {
    Err("rebuild with `--features metal` to use the Burn Metal backend".into())
}

fn smoke_with_backend<B: AutodiffBackend>(
    args: SmokeArgs,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    let mut config = args.model.config();
    config.backbone.head_upsample_mode =
        resolve_head_upsample(args.backend, args.head_upsample_mode);

    let _model = run_synthetic_training::<B>(
        config,
        device,
        args.steps,
        args.batch_size,
        args.input_size.height,
        args.input_size.width,
        args.learning_rate,
    );
    Ok(())
}

fn resolve_head_upsample(
    backend: BackendArg,
    requested: Option<HeadUpsampleArg>,
) -> HeadUpsampleMode {
    requested.map(HeadUpsampleArg::mode).unwrap_or_else(|| {
        if backend == BackendArg::Metal {
            HeadUpsampleMode::Nearest
        } else {
            HeadUpsampleMode::BilinearAligned
        }
    })
}

fn limited_len(len: usize, limit: Option<usize>) -> usize {
    limit.map_or(len, |limit| len.min(limit))
}

fn print_train_start(
    args: &TrainArgs,
    train_samples: usize,
    val_samples: Option<usize>,
    head_upsample_mode: HeadUpsampleMode,
) {
    println!("Training COCO keypoints");
    println!("  backend: {}", args.backend);
    println!("  model: {}", args.model);
    println!(
        "  input: {}x{}  batch: {}  epochs: {}  lr: {}",
        args.input_size.height,
        args.input_size.width,
        args.batch_size,
        args.epochs,
        args.learning_rate
    );
    println!(
        "  head upsample: {}",
        format_head_upsample(head_upsample_mode)
    );
    println!(
        "  train: {} samples ({})",
        train_samples,
        args.train_ann.display()
    );
    match (&args.val_ann, val_samples) {
        (Some(val_ann), Some(samples)) => {
            println!("  val: {} samples ({})", samples, val_ann.display());
        }
        _ => println!("  val: disabled"),
    }
    println!("  checkpoints: {}", args.out_dir.display());
    if args.log_every > 0 {
        println!("  progress: every {} batches", args.log_every);
    } else {
        println!("  progress: epoch only");
    }
}

fn print_training_progress(progress: PoseTrainingProgress, total_epochs: usize) {
    match progress {
        PoseTrainingProgress::Batch(batch) => {
            println!(
                "[epoch {}] batch {:05} samples {} train_loss {:.6}",
                format_epoch(batch.epoch, total_epochs),
                batch.train_batches,
                batch.train_samples,
                batch.train_loss
            );
        }
        PoseTrainingProgress::Epoch(epoch) => {
            println!(
                "[epoch {}] train_loss {:.6} val_loss {} elapsed {:.2}s",
                format_epoch(epoch.epoch, total_epochs),
                epoch.train_loss,
                format_optional_loss(epoch.val_loss),
                epoch.elapsed_seconds
            );
        }
    }
}

fn print_training_done(report: &PoseTrainingReport, checkpoint_dir: &PathBuf) {
    let last = report.epochs.last().expect("at least one epoch");
    println!("Finished training");
    println!("  final epoch: {}", last.epoch);
    println!("  train loss: {:.6}", last.train_loss);
    println!("  val loss: {}", format_optional_loss(last.val_loss));
    if let Some(best_val_loss) = report.best_val_loss {
        println!("  best val loss: {best_val_loss:.6}");
    }
    println!(
        "  report: {}",
        checkpoint_dir.join("training_report.json").display()
    );
    println!(
        "  last checkpoint: {}",
        checkpoint_dir.join("last.mpk").display()
    );
}

fn print_smoke_start(args: &SmokeArgs) {
    let head_upsample_mode = resolve_head_upsample(args.backend, args.head_upsample_mode);
    println!("Running synthetic smoke check");
    println!("  backend: {}", args.backend);
    println!("  model: {}", args.model);
    println!(
        "  input: {}x{}  batch: {}  steps: {}  lr: {}",
        args.input_size.height,
        args.input_size.width,
        args.batch_size,
        args.steps,
        args.learning_rate
    );
    println!(
        "  head upsample: {}",
        format_head_upsample(head_upsample_mode)
    );
}

fn print_smoke_done(backend: BackendArg) {
    println!("Finished synthetic smoke check ({backend})");
}

fn format_head_upsample(mode: HeadUpsampleMode) -> &'static str {
    match mode {
        HeadUpsampleMode::BilinearAligned => "bilinear",
        HeadUpsampleMode::Nearest => "nearest",
    }
}

fn format_optional_loss(loss: Option<f64>) -> String {
    loss.map(|loss| format!("{loss:.6}"))
        .unwrap_or_else(|| "n/a".to_string())
}

fn format_epoch(epoch: usize, total_epochs: usize) -> String {
    let width = total_epochs.to_string().len().max(3);
    format!("{:0width$}/{total_epochs}", epoch, width = width)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_train_command() {
        let cli = Cli::parse_from([
            "lite-hrnet-burn",
            "train",
            "--annotations",
            "person_keypoints_train.json",
            "--images",
            "train2017",
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
        assert_eq!(args.input_size.height, 128);
        assert_eq!(args.input_size.width, 96);
        assert_eq!(args.model, ModelArg::LiteHrNet30);
    }

    #[test]
    fn parses_smoke_command() {
        let cli = Cli::parse_from([
            "lite-hrnet-burn",
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
    fn rejects_invalid_input_size() {
        let error = Cli::try_parse_from(["lite-hrnet-burn", "smoke", "--input-size", "64"])
            .expect_err("invalid input size should fail");

        assert!(error.to_string().contains("HEIGHTxWIDTH"));
    }

    #[test]
    fn rejects_validation_images_without_validation_annotations() {
        let error = Cli::try_parse_from([
            "lite-hrnet-burn",
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
