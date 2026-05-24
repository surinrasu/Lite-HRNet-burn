use std::{
    error::Error,
    path::{Path, PathBuf},
};

use ann::{backend::Autodiff, tensor::backend::AutodiffBackend};
use pose_obc_retrieval::{
    CocoPoseDataset, HeadUpsampleMode, PoseDataConfig, PoseTrainingConfig, PoseTrainingProgress,
    PoseTrainingReport,
    train::{run_synthetic_training, train_dataset_with_progress},
};

use super::args::{BackendArg, HeadUpsampleArg, SmokeArgs, TrainArgs};

#[cfg(feature = "metal")]
use super::init_metal_device;

pub(super) fn run_train(args: TrainArgs) -> Result<(), Box<dyn Error>> {
    match args.backend {
        BackendArg::Flex => {
            type Backend = Autodiff<ann::backend::Flex>;
            let device = Default::default();
            train_with_backend::<Backend>(args, &device)
        }
        BackendArg::Metal => train_metal(args),
    }
}

#[cfg(feature = "metal")]
fn train_metal(args: TrainArgs) -> Result<(), Box<dyn Error>> {
    use ann::backend::Metal;

    type Backend = Autodiff<Metal>;
    let device = init_metal_device()?;
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
    let mut model = args.model.config();
    model.backbone.head_upsample_mode =
        resolve_head_upsample(args.backend, args.head_upsample_mode);

    let data = PoseDataConfig {
        sigma: args.sigma,
        ..PoseDataConfig::from_input(
            args.input_size.height,
            args.input_size.width,
            model.num_joints,
        )
    };

    let train_pose_dir = args
        .train_pose_dir
        .clone()
        .or_else(|| infer_spinepose_pose_dir(&args.train_images));
    let train_data = load_pose_dataset(
        &args.train_ann,
        &args.train_images,
        train_pose_dir.as_deref(),
        data.clone(),
    )?;
    let val_dataset = match &args.val_ann {
        Some(val_ann) => {
            let val_images = args.val_images.as_ref().unwrap_or(&args.train_images);
            let val_pose_dir = args
                .val_pose_dir
                .clone()
                .or_else(|| infer_spinepose_pose_dir(val_images));
            Some(load_pose_dataset(
                val_ann,
                val_images,
                val_pose_dir.as_deref(),
                data,
            )?)
        }
        None => None,
    };

    let train_samples = limited_len(train_data.len(), args.max_train_samples);
    let val_samples = val_dataset
        .as_ref()
        .map(|dataset| limited_len(dataset.len(), args.max_val_samples));
    let prefetch_batches = resolve_prefetch_batches(args.backend, args.prefetch_batches);
    print_train_start(
        &args,
        train_samples,
        val_samples,
        model.num_joints,
        train_pose_dir.as_deref(),
        model.backbone.head_upsample_mode,
        prefetch_batches,
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
        prefetch_batches,
    };

    let (_model, report) =
        train_dataset_with_progress::<B, _>(config, train_data, val_dataset, device, |progress| {
            print_training_progress(progress, total_epochs);
        })?;
    print_training_done(&report, &checkpoint_dir);
    Ok(())
}

pub(super) fn run_smoke(args: SmokeArgs) -> Result<(), Box<dyn Error>> {
    let backend = args.backend;
    match backend {
        BackendArg::Flex => {
            print_smoke_start(&args);
            type Backend = Autodiff<ann::backend::Flex>;
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
    use ann::backend::Metal;

    type Backend = Autodiff<Metal>;
    let device = init_metal_device()?;
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

fn resolve_prefetch_batches(backend: BackendArg, requested: Option<usize>) -> usize {
    requested.unwrap_or(match backend {
        BackendArg::Flex => 0,
        BackendArg::Metal => 2,
    })
}

fn load_pose_dataset(
    annotations: &Path,
    images: &Path,
    pose_dir: Option<&Path>,
    data: PoseDataConfig,
) -> Result<CocoPoseDataset, Box<dyn Error>> {
    match pose_dir {
        Some(pose_dir) => Ok(CocoPoseDataset::from_coco_with_spinepose(
            annotations,
            images,
            pose_dir,
            data,
        )?),
        None => Ok(CocoPoseDataset::from_coco(annotations, images, data)?),
    }
}

fn infer_spinepose_pose_dir(image_root: &Path) -> Option<PathBuf> {
    let parent = image_root.parent()?;
    let split = image_root.file_name()?;
    let candidate = parent.join("poses").join(split);
    candidate.is_dir().then_some(candidate)
}

fn limited_len(len: usize, limit: Option<usize>) -> usize {
    limit.map_or(len, |limit| len.min(limit))
}

fn print_train_start(
    args: &TrainArgs,
    train_samples: usize,
    val_samples: Option<usize>,
    num_joints: usize,
    train_pose_dir: Option<&Path>,
    head_upsample_mode: HeadUpsampleMode,
    prefetch_batches: usize,
) {
    println!("Training COCO SpinePose keypoints");
    println!("  backend: {}", args.backend);
    println!("  model: {}", args.model);
    println!("  keypoints: {num_joints}");
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
    match train_pose_dir {
        Some(pose_dir) => println!("  train poses: {}", pose_dir.display()),
        None => println!("  train poses: COCO annotation keypoints"),
    }
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
    println!("  prefetch: {prefetch_batches} CPU batches");
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

fn print_training_done(report: &PoseTrainingReport, checkpoint_dir: &Path) {
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
