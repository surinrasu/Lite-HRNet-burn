use std::collections::HashSet;
use std::path::{Path, PathBuf};

use ann::tensor::backend::Backend;
use serde::{Deserialize, Serialize};

use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateIndex {
    pub version: u32,
    pub model: RetrievalModelConfig,
    pub entries: Vec<CandidateEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CandidateEntry {
    pub id: String,
    pub codepoint: Option<String>,
    pub character: Option<String>,
    pub persona: String,
    pub glyph_path: PathBuf,
    pub embedding: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct SearchHit {
    pub index: usize,
    pub entry: CandidateEntry,
    pub score: f32,
}

pub fn build_candidate_index<B: Backend>(
    model: &RetrievalModel<B>,
    model_config: RetrievalModelConfig,
    dataset: &RetrievalPairDataset,
    unique_by_id: bool,
    device: &B::Device,
) -> Result<CandidateIndex, RetrievalError> {
    let mut entries = Vec::new();
    let mut seen_paths = HashSet::new();

    for candidate in dataset.glyph_candidates(unique_by_id) {
        if !seen_paths.insert(candidate.glyph_path.clone()) {
            continue;
        }
        let features = extract_glyph_features_from_path(&candidate.glyph_path)?;
        let embedding = encode_glyph_features(model, &features, device)?;
        entries.push(CandidateEntry {
            id: candidate.id,
            codepoint: candidate.codepoint,
            character: candidate.character,
            persona: candidate.persona,
            glyph_path: candidate.glyph_path,
            embedding,
        });
    }

    if entries.is_empty() {
        return Err(RetrievalError::InvalidData(
            "candidate index would be empty".to_string(),
        ));
    }

    Ok(CandidateIndex {
        version: CANDIDATE_INDEX_VERSION,
        model: model_config,
        entries,
    })
}

pub fn write_candidate_index(
    path: impl AsRef<Path>,
    index: &CandidateIndex,
) -> Result<(), RetrievalError> {
    validate_candidate_index(index)?;
    write_json_file(path, index)
}

pub fn read_candidate_index(path: impl AsRef<Path>) -> Result<CandidateIndex, RetrievalError> {
    let index = read_json_file(path)?;
    validate_candidate_index(&index)?;
    Ok(index)
}

pub fn search_index(
    index: &CandidateIndex,
    query_embedding: &[f32],
    top_k: usize,
) -> Result<Vec<SearchHit>, RetrievalError> {
    validate_candidate_index(index)?;
    validate_embedding(
        "query embedding",
        query_embedding,
        index.model.embedding_dim,
        None,
    )?;

    let mut hits = index
        .entries
        .iter()
        .enumerate()
        .map(|(entry_index, entry)| SearchHit {
            index: entry_index,
            entry: entry.clone(),
            score: dot(query_embedding, &entry.embedding),
        })
        .collect::<Vec<_>>();
    hits.sort_by(|left, right| right.score.total_cmp(&left.score));
    hits.truncate(top_k.min(hits.len()));
    Ok(hits)
}

pub fn validate_candidate_index(index: &CandidateIndex) -> Result<(), RetrievalError> {
    if index.version != CANDIDATE_INDEX_VERSION {
        return Err(RetrievalError::InvalidData(format!(
            "unsupported candidate index version: expected {CANDIDATE_INDEX_VERSION}, got {}",
            index.version
        )));
    }
    validate_model_config(&index.model)?;
    if index.entries.is_empty() {
        return Err(RetrievalError::InvalidData(
            "candidate index has no entries".to_string(),
        ));
    }
    for (entry_index, entry) in index.entries.iter().enumerate() {
        validate_embedding(
            "candidate embedding",
            &entry.embedding,
            index.model.embedding_dim,
            Some(entry_index),
        )?;
    }
    Ok(())
}

fn validate_model_config(config: &RetrievalModelConfig) -> Result<(), RetrievalError> {
    if config.input_dim == 0 {
        return Err(RetrievalError::InvalidData(
            "retrieval model input_dim must be greater than zero".to_string(),
        ));
    }
    if config.hidden_dim == 0 {
        return Err(RetrievalError::InvalidData(
            "retrieval model hidden_dim must be greater than zero".to_string(),
        ));
    }
    if config.embedding_dim == 0 {
        return Err(RetrievalError::InvalidData(
            "retrieval model embedding_dim must be greater than zero".to_string(),
        ));
    }
    Ok(())
}

fn validate_embedding(
    label: &str,
    embedding: &[f32],
    expected_dim: usize,
    entry_index: Option<usize>,
) -> Result<(), RetrievalError> {
    if embedding.len() != expected_dim {
        let where_text = entry_index
            .map(|index| format!(" at entry {index}"))
            .unwrap_or_default();
        return Err(RetrievalError::InvalidData(format!(
            "{label}{where_text} dimension mismatch: expected {expected_dim}, got {}",
            embedding.len()
        )));
    }
    if let Some(value_index) = embedding.iter().position(|value| !value.is_finite()) {
        let where_text = entry_index
            .map(|index| format!(" at entry {index}"))
            .unwrap_or_default();
        return Err(RetrievalError::InvalidData(format!(
            "{label}{where_text} value {value_index} is not finite"
        )));
    }
    Ok(())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| left * right)
        .sum()
}
