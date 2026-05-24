use std::collections::BTreeMap;

use ann::tensor::backend::Backend;

use crate::{RetrievalError, SearchHit, encode_pose_features, search_index};

use super::{
    RetrievalService,
    http::{HttpRequest, multipart_boundary, parse_multipart, parse_top_k, parse_urlencoded},
    live::{LiveSearchResponse, live_search_response},
};

pub(super) fn sample_search_from_query(
    service: &RetrievalService<impl Backend>,
    query: &BTreeMap<String, String>,
) -> Result<(Vec<SearchHit>, String, usize), RetrievalError> {
    let sample = query
        .get("sample")
        .ok_or_else(|| RetrievalError::InvalidData("missing sample query parameter".to_string()))?
        .parse::<usize>()
        .map_err(|_| {
            RetrievalError::InvalidData("sample must be a non-negative integer".to_string())
        })?;
    let top_k = parse_top_k(query.get("k"), service.default_top_k);
    let pair = service.dataset.pairs().get(sample).ok_or_else(|| {
        RetrievalError::InvalidData(format!("sample index {sample} out of range"))
    })?;
    let features = service
        .pose_estimator
        .estimate_pose_features_from_path(&pair.image_path)?;
    let embedding = encode_pose_features(&service.model, &features, &service.device)?;
    let hits = search_index(&service.index, &embedding, top_k)?;
    Ok((hits, format!("sample #{sample} {}", pair.id), top_k))
}

pub(super) fn upload_search(
    service: &RetrievalService<impl Backend>,
    request: &HttpRequest,
) -> Result<(Vec<SearchHit>, String, usize), RetrievalError> {
    let content_type = request
        .headers
        .get("content-type")
        .map(String::as_str)
        .unwrap_or("");

    if let Some(boundary) = multipart_boundary(content_type) {
        let form = parse_multipart(&request.body, &boundary)?;
        let top_k = parse_top_k(form.fields.get("k"), service.default_top_k);
        if let Some(sample) = form
            .fields
            .get("sample")
            .filter(|value| !value.trim().is_empty())
        {
            let mut query = BTreeMap::new();
            query.insert("sample".to_string(), sample.clone());
            query.insert("k".to_string(), top_k.to_string());
            return sample_search_from_query(service, &query);
        }

        let image = form.files.get("image").ok_or_else(|| {
            RetrievalError::InvalidData("upload an image or provide a sample id".to_string())
        })?;
        if image.is_empty() {
            return Err(RetrievalError::InvalidData(
                "uploaded image is empty".to_string(),
            ));
        }
        let features = service
            .pose_estimator
            .estimate_pose_features_from_bytes(image)?;
        let embedding = encode_pose_features(&service.model, &features, &service.device)?;
        let hits = search_index(&service.index, &embedding, top_k)?;
        return Ok((hits, "uploaded image".to_string(), top_k));
    }

    let body = String::from_utf8_lossy(&request.body);
    let form = parse_urlencoded(&body);
    sample_search_from_query(service, &form)
}

pub(super) fn live_search(
    service: &RetrievalService<impl Backend>,
    request: &HttpRequest,
) -> Result<LiveSearchResponse, RetrievalError> {
    let top_k = parse_top_k(request.query.get("k"), service.default_top_k);
    let content_type = request
        .headers
        .get("content-type")
        .map(String::as_str)
        .unwrap_or("");

    let frame = if let Some(boundary) = multipart_boundary(content_type) {
        let form = parse_multipart(&request.body, &boundary)?;
        form.files
            .get("frame")
            .or_else(|| form.files.get("image"))
            .cloned()
            .ok_or_else(|| {
                RetrievalError::InvalidData(
                    "live frame request must include a frame or image file".to_string(),
                )
            })?
    } else {
        request.body.clone()
    };

    if frame.is_empty() {
        return Err(RetrievalError::InvalidData(
            "live frame request body is empty".to_string(),
        ));
    }

    let features = service
        .pose_estimator
        .estimate_pose_features_from_bytes(&frame)?;
    let embedding = encode_pose_features(&service.model, &features, &service.device)?;
    let hits = search_index(&service.index, &embedding, top_k)?;
    Ok(live_search_response(&hits, top_k))
}
