use std::path::PathBuf;

use cli::{ArgAction, Args};

use super::{BackendArg, parse_positive_count, parse_positive_f64};

extern crate cli as clap;

#[derive(Clone, Debug, Args)]
pub(in crate::cli) struct RetrievalTrainArgs {
    /// Burn backend to use.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    pub(in crate::cli) backend: BackendArg,
    /// Root containing data/persona_*/images and data/persona_*/glyphs.
    #[arg(long = "data-root", value_name = "DIR", default_value = "data")]
    pub(in crate::cli) data_root: PathBuf,
    /// Directory for retrieval checkpoints, config, and training report.
    #[arg(
        short = 'o',
        long = "output-dir",
        value_name = "DIR",
        default_value = "runs/retrieval"
    )]
    pub(in crate::cli) out_dir: PathBuf,
    /// Training epochs.
    #[arg(short = 'e', long, default_value_t = 20, value_parser = parse_positive_count)]
    pub(in crate::cli) epochs: usize,
    /// Batch size.
    #[arg(short = 'b', long, default_value_t = 32, value_parser = parse_positive_count)]
    pub(in crate::cli) batch_size: usize,
    /// Adam learning rate.
    #[arg(long = "learning-rate", value_name = "LR", default_value_t = 1e-3, value_parser = parse_positive_f64)]
    pub(in crate::cli) learning_rate: f64,
    /// Hidden dimension of each MLP tower.
    #[arg(long = "hidden-dim", default_value_t = 128, value_parser = parse_positive_count)]
    pub(in crate::cli) hidden_dim: usize,
    /// Shared embedding dimension used for cosine search.
    #[arg(long = "embedding-dim", default_value_t = 64, value_parser = parse_positive_count)]
    pub(in crate::cli) embedding_dim: usize,
    /// Contrastive softmax temperature.
    #[arg(long, default_value_t = 0.07, value_parser = parse_positive_f64)]
    pub(in crate::cli) temperature: f64,
    /// Limit the number of paired training samples.
    #[arg(long = "max-pairs", value_name = "N", value_parser = parse_positive_count)]
    pub(in crate::cli) max_pairs: Option<usize>,
    /// Print batch progress every N batches. Use 0 for epoch-only progress.
    #[arg(long, default_value_t = 20)]
    pub(in crate::cli) log_every: usize,
    /// RNG seed used for sample shuffling.
    #[arg(long, default_value_t = 42)]
    pub(in crate::cli) seed: u64,
    /// Disable dataset shuffling.
    #[arg(long = "no-shuffle", action = ArgAction::SetFalse, default_value_t = true)]
    pub(in crate::cli) shuffle: bool,
    /// Save epoch_NNN.mpk checkpoints in addition to last.mpk.
    #[arg(long = "save-every-epoch", action = ArgAction::SetTrue)]
    pub(in crate::cli) save_every_epoch: bool,
}

#[derive(Clone, Debug, Args)]
pub(in crate::cli) struct RetrievalIndexArgs {
    /// Burn backend to use for candidate encoding.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    pub(in crate::cli) backend: BackendArg,
    /// Root containing data/persona_*/images and data/persona_*/glyphs.
    #[arg(long = "data-root", value_name = "DIR", default_value = "data")]
    pub(in crate::cli) data_root: PathBuf,
    /// Retrieval model checkpoint, with or without the .mpk extension.
    #[arg(long, value_name = "PATH", default_value = "runs/retrieval/last.mpk")]
    pub(in crate::cli) model: PathBuf,
    /// Retrieval model config JSON. Defaults to retrieval_config.json next to --model.
    #[arg(long, value_name = "PATH")]
    pub(in crate::cli) config: Option<PathBuf>,
    /// Output candidate embedding index.
    #[arg(
        short = 'o',
        long,
        value_name = "PATH",
        default_value = "runs/retrieval/glyph_index.json"
    )]
    pub(in crate::cli) output: PathBuf,
    /// Keep duplicate glyph candidates from different persona directories.
    #[arg(long = "include-duplicate-glyphs", action = ArgAction::SetFalse, default_value_t = true)]
    pub(in crate::cli) unique_glyphs: bool,
}

#[derive(Clone, Debug, Args)]
pub(in crate::cli) struct RetrievalSearchArgs {
    /// Burn backend to use for query encoding.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    pub(in crate::cli) backend: BackendArg,
    /// Candidate embedding index JSON.
    #[arg(
        long,
        value_name = "PATH",
        default_value = "runs/retrieval/glyph_index.json"
    )]
    pub(in crate::cli) index: PathBuf,
    /// Retrieval model checkpoint, with or without the .mpk extension.
    #[arg(long, value_name = "PATH", default_value = "runs/retrieval/last.mpk")]
    pub(in crate::cli) model: PathBuf,
    /// Retrieval model config JSON. Defaults to config stored in --index or next to --model.
    #[arg(long, value_name = "PATH")]
    pub(in crate::cli) config: Option<PathBuf>,
    /// Root containing data/persona_*/images and data/persona_*/glyphs, used with --sample.
    #[arg(long = "data-root", value_name = "DIR", default_value = "data")]
    pub(in crate::cli) data_root: PathBuf,
    /// Query image path.
    #[arg(long, value_name = "PATH", conflicts_with = "sample")]
    pub(in crate::cli) image: Option<PathBuf>,
    /// Query by pair index from the data directory.
    #[arg(long, value_name = "N", conflicts_with = "image")]
    pub(in crate::cli) sample: Option<usize>,
    /// Number of hits to return.
    #[arg(short = 'k', long = "top-k", default_value_t = 8, value_parser = parse_positive_count)]
    pub(in crate::cli) top_k: usize,
}

#[derive(Clone, Debug, Args)]
pub(in crate::cli) struct RetrievalServeArgs {
    /// Burn backend to use for query and candidate encoding.
    #[arg(long, value_enum, default_value_t = BackendArg::Flex)]
    pub(in crate::cli) backend: BackendArg,
    /// Address for the HTTP UI.
    #[arg(long, default_value = "127.0.0.1:8080")]
    pub(in crate::cli) addr: String,
    /// Root containing data/persona_*/images and data/persona_*/glyphs.
    #[arg(long = "data-root", value_name = "DIR", default_value = "data")]
    pub(in crate::cli) data_root: PathBuf,
    /// Optional precomputed candidate embedding index JSON.
    #[arg(long, value_name = "PATH")]
    pub(in crate::cli) index: Option<PathBuf>,
    /// Retrieval model checkpoint, with or without the .mpk extension.
    #[arg(long, value_name = "PATH", default_value = "runs/retrieval/last.mpk")]
    pub(in crate::cli) model: PathBuf,
    /// Retrieval model config JSON. Defaults to config stored in --index or next to --model.
    #[arg(long, value_name = "PATH")]
    pub(in crate::cli) config: Option<PathBuf>,
    /// Default number of hits in the UI.
    #[arg(short = 'k', long = "top-k", default_value_t = 8, value_parser = parse_positive_count)]
    pub(in crate::cli) top_k: usize,
    /// Enable live browser video frame retrieval/scoring.
    #[arg(long, action = ArgAction::SetTrue)]
    pub(in crate::cli) live: bool,
    /// Keep duplicate glyph candidates from different persona directories when building in memory.
    #[arg(long = "include-duplicate-glyphs", action = ArgAction::SetFalse, default_value_t = true)]
    pub(in crate::cli) unique_glyphs: bool,
}
