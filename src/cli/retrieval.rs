use std::{
    error::Error,
    path::{Path, PathBuf},
};

use ann::{
    backend::Autodiff,
    tensor::backend::{AutodiffBackend, Backend},
};
use pose_obc_retrieval::{
    CandidateIndex, DefaultPoseEstimator, RetrievalError, RetrievalModelConfig,
    RetrievalPairDataset, RetrievalTrainingConfig, RetrievalTrainingProgress,
    build_candidate_index_with_cache, encode_pose_features, extract_pose_features_from_path,
    load_retrieval_model, load_retrieval_model_config, read_candidate_index, search_index,
    serve_retrieval, service::RetrievalService, train_retrieval_dataset, write_candidate_index,
};

use super::args::{
    BackendArg, RetrievalIndexArgs, RetrievalSearchArgs, RetrievalServeArgs, RetrievalTrainArgs,
};

#[cfg(feature = "metal")]
use super::init_metal_device;

pub(super) fn run_retrieval_train(args: RetrievalTrainArgs) -> Result<(), Box<dyn Error>> {
    match args.backend {
        BackendArg::Flex => {
            type Backend = Autodiff<ann::backend::Flex>;
            let device = Default::default();
            run_retrieval_train_with_backend::<Backend>(args, &device)
        }
        BackendArg::Metal => run_retrieval_train_metal(args),
    }
}

#[cfg(feature = "metal")]
fn run_retrieval_train_metal(args: RetrievalTrainArgs) -> Result<(), Box<dyn Error>> {
    use ann::backend::Metal;

    type Backend = Autodiff<Metal>;
    let device = init_metal_device()?;
    run_retrieval_train_with_backend::<Backend>(args, &device)
}

#[cfg(not(feature = "metal"))]
fn run_retrieval_train_metal(_args: RetrievalTrainArgs) -> Result<(), Box<dyn Error>> {
    Err("rebuild with `--features metal` to train retrieval on Metal".into())
}

fn run_retrieval_train_with_backend<B: AutodiffBackend>(
    args: RetrievalTrainArgs,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    let dataset = RetrievalPairDataset::from_data_root(&args.data_root)?;
    let model = RetrievalModelConfig {
        input_dim: pose_obc_retrieval::RETRIEVAL_FEATURE_DIM,
        hidden_dim: args.hidden_dim,
        embedding_dim: args.embedding_dim,
    };
    let config = RetrievalTrainingConfig {
        model,
        epochs: args.epochs,
        batch_size: args.batch_size,
        learning_rate: args.learning_rate,
        temperature: args.temperature,
        shuffle: args.shuffle,
        seed: args.seed,
        max_pairs: args.max_pairs,
        checkpoint_dir: args.out_dir.clone(),
        log_every: args.log_every,
        save_every_epoch: args.save_every_epoch,
    };

    let train_pairs = limited_len(dataset.len(), args.max_pairs);
    println!("Training pose/glyph retrieval");
    println!("  backend: {}", args.backend);
    println!("  pairs: {train_pairs} ({})", args.data_root.display());
    println!(
        "  input: {}  hidden: {}  embedding: {}",
        config.model.input_dim, config.model.hidden_dim, config.model.embedding_dim
    );
    println!(
        "  batch: {}  epochs: {}  lr: {}  temperature: {}",
        config.batch_size, config.epochs, config.learning_rate, config.temperature
    );
    println!("  checkpoints: {}", args.out_dir.display());

    let checkpoint_dir = args.out_dir.clone();
    let total_epochs = args.epochs;
    let (_model, report) = train_retrieval_dataset::<B, _>(config, &dataset, device, |progress| {
        print_retrieval_progress(progress, total_epochs);
    })?;
    let last = report.epochs.last().expect("at least one epoch");
    println!("Finished retrieval training");
    println!("  final epoch: {}", last.epoch);
    println!("  train loss: {:.6}", last.train_loss);
    println!(
        "  config: {}",
        checkpoint_dir.join("retrieval_config.json").display()
    );
    println!(
        "  last checkpoint: {}",
        checkpoint_dir.join("last.mpk").display()
    );
    Ok(())
}

pub(super) fn run_retrieval_index(args: RetrievalIndexArgs) -> Result<(), Box<dyn Error>> {
    match args.backend {
        BackendArg::Flex => {
            type Backend = ann::backend::Flex;
            let device = Default::default();
            run_retrieval_index_with_backend::<Backend>(args, &device)
        }
        BackendArg::Metal => run_retrieval_index_metal(args),
    }
}

#[cfg(feature = "metal")]
fn run_retrieval_index_metal(args: RetrievalIndexArgs) -> Result<(), Box<dyn Error>> {
    use ann::backend::Metal;

    type Backend = Metal;
    let device = init_metal_device()?;
    run_retrieval_index_with_backend::<Backend>(args, &device)
}

#[cfg(not(feature = "metal"))]
fn run_retrieval_index_metal(_args: RetrievalIndexArgs) -> Result<(), Box<dyn Error>> {
    Err("rebuild with `--features metal` to index retrieval candidates on Metal".into())
}

fn run_retrieval_index_with_backend<B: Backend>(
    args: RetrievalIndexArgs,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    let dataset = RetrievalPairDataset::from_data_root(&args.data_root)?;
    let model_config =
        load_retrieval_config_or_default(args.config.as_deref(), Some(args.model.as_path()), None)?;
    let model = load_retrieval_model::<B>(&model_config, &args.model, device)?;
    let feature_cache_dir = feature_cache_dir_for_output(&args.output);
    let index = build_candidate_index_with_cache(
        &model,
        model_config,
        &dataset,
        args.unique_glyphs,
        device,
        Some(&feature_cache_dir),
    )?;
    ensure_parent_dir(&args.output)?;
    write_candidate_index(&args.output, &index)?;
    println!("Built candidate glyph index");
    println!("  backend: {}", args.backend);
    println!("  candidates: {}", index.entries.len());
    println!("  output: {}", args.output.display());
    Ok(())
}

pub(super) fn run_retrieval_search(args: RetrievalSearchArgs) -> Result<(), Box<dyn Error>> {
    match args.backend {
        BackendArg::Flex => {
            type Backend = ann::backend::Flex;
            let device = Default::default();
            run_retrieval_search_with_backend::<Backend>(args, &device)
        }
        BackendArg::Metal => run_retrieval_search_metal(args),
    }
}

#[cfg(feature = "metal")]
fn run_retrieval_search_metal(args: RetrievalSearchArgs) -> Result<(), Box<dyn Error>> {
    use ann::backend::Metal;

    type Backend = Metal;
    let device = init_metal_device()?;
    run_retrieval_search_with_backend::<Backend>(args, &device)
}

#[cfg(not(feature = "metal"))]
fn run_retrieval_search_metal(_args: RetrievalSearchArgs) -> Result<(), Box<dyn Error>> {
    Err("rebuild with `--features metal` to search retrieval on Metal".into())
}

fn run_retrieval_search_with_backend<B: Backend>(
    args: RetrievalSearchArgs,
    device: &B::Device,
) -> Result<(), Box<dyn Error>> {
    let index = read_candidate_index(&args.index)?;
    let model_config = load_retrieval_config_or_default(
        args.config.as_deref(),
        Some(args.model.as_path()),
        Some(&index),
    )?;
    let model = load_retrieval_model::<B>(&model_config, &args.model, device)?;

    let (features, label) = match (&args.image, args.sample) {
        (Some(image), None) => (
            extract_pose_features_from_path(image)?,
            image.display().to_string(),
        ),
        (None, Some(sample)) => {
            let dataset = RetrievalPairDataset::from_data_root(&args.data_root)?;
            let pair = dataset.pairs().get(sample).ok_or_else(|| {
                RetrievalError::InvalidData(format!(
                    "sample index {sample} out of range for {} pairs",
                    dataset.len()
                ))
            })?;
            (
                extract_pose_features_from_path(&pair.image_path)?,
                format!("sample #{sample} {}", pair.id),
            )
        }
        _ => {
            return Err("provide exactly one of --image or --sample".into());
        }
    };
    let embedding = encode_pose_features(&model, &features, device)?;
    let hits = search_index(&index, &embedding, args.top_k)?;

    println!("Query: {label}");
    println!("Backend: {}", args.backend);
    for (rank, hit) in hits.iter().enumerate() {
        let label = hit
            .entry
            .character
            .as_deref()
            .unwrap_or(hit.entry.id.as_str());
        println!(
            "{:02}. score {:.4}  {}  {}  {}",
            rank + 1,
            hit.score,
            label,
            hit.entry.codepoint.as_deref().unwrap_or("-"),
            hit.entry.glyph_path.display()
        );
    }
    Ok(())
}

pub(super) fn run_retrieval_serve(args: RetrievalServeArgs) -> Result<(), Box<dyn Error>> {
    match args.backend {
        BackendArg::Flex => {
            type Backend = ann::backend::Flex;
            let device = Default::default();
            run_retrieval_serve_with_backend::<Backend>(args, device)
        }
        BackendArg::Metal => run_retrieval_serve_metal(args),
    }
}

#[cfg(feature = "metal")]
fn run_retrieval_serve_metal(args: RetrievalServeArgs) -> Result<(), Box<dyn Error>> {
    use ann::backend::Metal;

    type Backend = Metal;
    let device = init_metal_device()?;
    run_retrieval_serve_with_backend::<Backend>(args, device)
}

#[cfg(not(feature = "metal"))]
fn run_retrieval_serve_metal(_args: RetrievalServeArgs) -> Result<(), Box<dyn Error>> {
    Err("rebuild with `--features metal` to serve retrieval on Metal".into())
}

fn run_retrieval_serve_with_backend<B: Backend>(
    args: RetrievalServeArgs,
    device: B::Device,
) -> Result<(), Box<dyn Error>> {
    let dataset = RetrievalPairDataset::from_data_root(&args.data_root)?;

    let (index, model_config) = match &args.index {
        Some(index_path) => {
            let index = read_candidate_index(index_path)?;
            let model_config = load_retrieval_config_or_default(
                args.config.as_deref(),
                Some(args.model.as_path()),
                Some(&index),
            )?;
            (index, model_config)
        }
        None => {
            let model_config = load_retrieval_config_or_default(
                args.config.as_deref(),
                Some(args.model.as_path()),
                None,
            )?;
            let model = load_retrieval_model::<B>(&model_config, &args.model, &device)?;
            let feature_cache_dir = feature_cache_dir_for_model(&args.model);
            let index = build_candidate_index_with_cache(
                &model,
                model_config.clone(),
                &dataset,
                args.unique_glyphs,
                &device,
                Some(&feature_cache_dir),
            )?;
            (index, model_config)
        }
    };

    let model = load_retrieval_model::<B>(&model_config, &args.model, &device)?;
    println!("Serving retrieval UI");
    println!("  backend: {}", args.backend);
    println!("  pairs: {}", dataset.len());
    println!("  candidates: {}", index.entries.len());
    println!("  model: {}", args.model.display());
    println!("  live: {}", if args.live { "enabled" } else { "disabled" });
    serve_retrieval(
        &args.addr,
        RetrievalService {
            model,
            pose_estimator: DefaultPoseEstimator::default(),
            index,
            dataset,
            device,
            default_top_k: args.top_k,
            live: args.live,
        },
    )?;
    Ok(())
}

fn print_retrieval_progress(progress: RetrievalTrainingProgress, total_epochs: usize) {
    match progress {
        RetrievalTrainingProgress::Batch(batch) => {
            println!(
                "[retrieval epoch {}] batch {:05} pairs {} train_loss {:.6}",
                format_epoch(batch.epoch, total_epochs),
                batch.train_batches,
                batch.train_pairs,
                batch.train_loss
            );
        }
        RetrievalTrainingProgress::Epoch(epoch) => {
            println!(
                "[retrieval epoch {}] train_loss {:.6} elapsed {:.2}s",
                format_epoch(epoch.epoch, total_epochs),
                epoch.train_loss,
                epoch.elapsed_seconds
            );
        }
    }
}

fn load_retrieval_config_or_default(
    config_path: Option<&Path>,
    model_path: Option<&Path>,
    index: Option<&CandidateIndex>,
) -> Result<RetrievalModelConfig, Box<dyn Error>> {
    if let Some(config_path) = config_path {
        return Ok(load_retrieval_model_config(config_path)?);
    }
    if let Some(index) = index {
        return Ok(index.model.clone());
    }
    if let Some(model_path) = model_path {
        let inferred = default_retrieval_config_path(model_path);
        if inferred.is_file() {
            return Ok(load_retrieval_model_config(inferred)?);
        }
    }
    Err("retrieval config not found; pass --config or use a model directory with retrieval_config.json".into())
}

fn default_retrieval_config_path(model_path: &Path) -> PathBuf {
    let base = if model_path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mpk"))
    {
        model_path.with_extension("")
    } else {
        model_path.to_path_buf()
    };
    base.parent()
        .map(|parent| parent.join("retrieval_config.json"))
        .unwrap_or_else(|| PathBuf::from("retrieval_config.json"))
}

fn ensure_parent_dir(path: &Path) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn feature_cache_dir_for_output(output: &Path) -> PathBuf {
    output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(|parent| parent.join("feature_cache"))
        .unwrap_or_else(|| PathBuf::from("feature_cache"))
}

fn feature_cache_dir_for_model(model: &Path) -> PathBuf {
    let model_stem = if model
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mpk"))
    {
        model.with_extension("")
    } else {
        model.to_path_buf()
    };
    model_stem
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .map(|parent| parent.join("feature_cache"))
        .unwrap_or_else(|| PathBuf::from("feature_cache"))
}

fn limited_len(len: usize, limit: Option<usize>) -> usize {
    limit.map_or(len, |limit| len.min(limit))
}

fn format_epoch(epoch: usize, total_epochs: usize) -> String {
    let width = total_epochs.to_string().len().max(3);
    format!("{:0width$}/{total_epochs}", epoch, width = width)
}
