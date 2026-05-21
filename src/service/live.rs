use serde::Serialize;

use crate::SearchHit;

#[derive(Debug, Serialize)]
pub(super) struct LiveSearchResponse {
    top_k: usize,
    hits: Vec<LiveSearchHit>,
}

#[derive(Debug, Serialize)]
struct LiveSearchHit {
    rank: usize,
    index: usize,
    id: String,
    character: Option<String>,
    codepoint: Option<String>,
    persona: String,
    score: f32,
    image_url: String,
}

pub(super) fn live_search_response(hits: &[SearchHit], top_k: usize) -> LiveSearchResponse {
    LiveSearchResponse {
        top_k,
        hits: hits
            .iter()
            .enumerate()
            .map(|(rank, hit)| LiveSearchHit {
                rank: rank + 1,
                index: hit.index,
                id: hit.entry.id.clone(),
                character: hit.entry.character.clone(),
                codepoint: hit.entry.codepoint.clone(),
                persona: hit.entry.persona.clone(),
                score: hit.score,
                image_url: format!("/candidate/{}", hit.index),
            })
            .collect(),
    }
}
