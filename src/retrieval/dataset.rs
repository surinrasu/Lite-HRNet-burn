use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};

use super::{RetrievalError, canonical_or_original};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RetrievalPair {
    pub id: String,
    pub codepoint: Option<String>,
    pub character: Option<String>,
    pub persona: String,
    pub image_path: PathBuf,
    pub glyph_path: PathBuf,
}

#[derive(Clone, Debug)]
pub struct RetrievalPairDataset {
    pairs: Vec<RetrievalPair>,
}

impl RetrievalPairDataset {
    pub fn from_data_root(data_root: impl AsRef<Path>) -> Result<Self, RetrievalError> {
        let data_root = resolve_existing_data_root(data_root.as_ref())?;
        let mut persona_dirs = Vec::new();

        for entry in fs::read_dir(&data_root)? {
            let entry = entry?;
            if !entry.file_type()?.is_dir() {
                continue;
            }
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("persona_") {
                persona_dirs.push((name, entry.path()));
            }
        }
        persona_dirs.sort_by(|left, right| left.0.cmp(&right.0));

        let mut pairs = Vec::new();
        for (persona, persona_dir) in persona_dirs {
            let image_dir = persona_dir.join("images");
            let glyph_dir = persona_dir.join("glyphs");
            if !image_dir.is_dir() || !glyph_dir.is_dir() {
                continue;
            }

            let mut image_paths = Vec::new();
            for entry in fs::read_dir(&image_dir)? {
                let entry = entry?;
                if entry.file_type()?.is_file() && is_png(&entry.path()) {
                    image_paths.push(entry.path());
                }
            }
            image_paths.sort();

            for image_path in image_paths {
                let Some(file_name) = image_path.file_name() else {
                    continue;
                };
                let glyph_path = glyph_dir.join(file_name);
                if !glyph_path.is_file() {
                    continue;
                }

                let Some(id) = image_path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .map(str::to_owned)
                else {
                    continue;
                };
                let metadata = parse_glyph_id(&id);
                pairs.push(RetrievalPair {
                    id,
                    codepoint: metadata.codepoint,
                    character: metadata.character,
                    persona: persona.clone(),
                    image_path: canonical_or_original(image_path),
                    glyph_path: canonical_or_original(glyph_path),
                });
            }
        }

        if pairs.is_empty() {
            return Err(RetrievalError::InvalidData(format!(
                "no image/glyph pairs found under {}",
                data_root.display()
            )));
        }

        Ok(Self { pairs })
    }

    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    pub fn pairs(&self) -> &[RetrievalPair] {
        &self.pairs
    }

    pub fn limited_pairs(&self, max_pairs: Option<usize>) -> Vec<RetrievalPair> {
        match max_pairs {
            Some(max_pairs) => self.pairs.iter().take(max_pairs).cloned().collect(),
            None => self.pairs.clone(),
        }
    }

    pub fn glyph_candidates(&self, unique_by_id: bool) -> Vec<GlyphCandidate> {
        if !unique_by_id {
            return self
                .pairs
                .iter()
                .map(GlyphCandidate::from_pair)
                .collect::<Vec<_>>();
        }

        let mut by_id = BTreeMap::new();
        for pair in &self.pairs {
            by_id
                .entry(pair.id.clone())
                .or_insert_with(|| GlyphCandidate::from_pair(pair));
        }
        by_id.into_values().collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GlyphCandidate {
    pub id: String,
    pub codepoint: Option<String>,
    pub character: Option<String>,
    pub persona: String,
    pub glyph_path: PathBuf,
}

impl GlyphCandidate {
    fn from_pair(pair: &RetrievalPair) -> Self {
        Self {
            id: pair.id.clone(),
            codepoint: pair.codepoint.clone(),
            character: pair.character.clone(),
            persona: pair.persona.clone(),
            glyph_path: pair.glyph_path.clone(),
        }
    }
}

pub fn resolve_existing_data_root(data_root: &Path) -> Result<PathBuf, RetrievalError> {
    if data_root.is_dir() {
        return Ok(canonical_or_original(data_root));
    }
    if data_root == Path::new("data") {
        let parent = Path::new("..").join("data");
        if parent.is_dir() {
            return Ok(canonical_or_original(parent));
        }
    }
    Err(RetrievalError::InvalidData(format!(
        "data root does not exist: {}",
        data_root.display()
    )))
}

fn is_png(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("png"))
}

struct GlyphIdMetadata {
    codepoint: Option<String>,
    character: Option<String>,
}

fn parse_glyph_id(id: &str) -> GlyphIdMetadata {
    let codepoint = id
        .split("_u")
        .nth(1)
        .and_then(|rest| rest.split('_').next())
        .map(str::to_owned);
    let character = codepoint
        .as_deref()
        .and_then(|hex| u32::from_str_radix(hex, 16).ok())
        .and_then(char::from_u32)
        .map(|ch| ch.to_string());
    GlyphIdMetadata {
        codepoint: codepoint.map(|value| format!("U+{}", value.to_uppercase())),
        character,
    }
}
