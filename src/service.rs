use std::{
    collections::BTreeMap,
    fs,
    net::{TcpListener, TcpStream},
    path::Path,
};

use ann::tensor::backend::Backend;

use crate::{
    CandidateIndex, DefaultPoseEstimator, RetrievalError, RetrievalModel, RetrievalPairDataset,
    SearchHit, encode_pose_features, search_index,
};

mod http;
mod live;
mod views;

use self::{
    http::{
        HttpRequest, HttpResponse, error_response, html_response, image_content_type,
        json_response, multipart_boundary, parse_multipart, parse_top_k, parse_urlencoded,
        read_request, static_response, write_response,
    },
    live::{LiveSearchResponse, live_search_response},
    views::{render_home, render_results},
};

const BEER_CSS: &[u8] = include_bytes!("../assets/beer.min.css");
const BEER_JS: &[u8] = include_bytes!("../assets/beer.min.js");
const MATERIAL_SYMBOLS_CSS: &[u8] = include_bytes!("../assets/material-symbols.css");
const MATERIAL_SYMBOLS_FONT: &[u8] = include_bytes!("../assets/material-symbols-outlined.ttf");
const EXAMPLE_ASSET_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/examples");
const EXAMPLE_ASSET_PREFIX: &str = "/assets/examples/";

pub struct RetrievalService<B: Backend> {
    pub model: RetrievalModel<B>,
    pub pose_estimator: DefaultPoseEstimator,
    pub index: CandidateIndex,
    pub dataset: RetrievalPairDataset,
    pub device: B::Device,
    pub default_top_k: usize,
    pub live: bool,
}

pub fn serve_retrieval<B: Backend>(
    addr: &str,
    service: RetrievalService<B>,
) -> Result<(), RetrievalError> {
    let listener = TcpListener::bind(addr)?;
    println!("Retrieval UI listening on http://{addr}");

    for stream in listener.incoming() {
        match stream {
            Ok(mut stream) => {
                if let Err(error) = handle_connection(&service, &mut stream) {
                    let response = error_response(500, &format!("internal error: {error}"));
                    let _ = write_response(&mut stream, response);
                }
            }
            Err(error) => eprintln!("failed to accept connection: {error}"),
        }
    }

    Ok(())
}

fn handle_connection<B: Backend>(
    service: &RetrievalService<B>,
    stream: &mut TcpStream,
) -> Result<(), RetrievalError> {
    let Some(request) = read_request(stream)? else {
        return Ok(());
    };
    let response = route_request(service, request);
    write_response(stream, response)?;
    Ok(())
}

fn route_request<B: Backend>(service: &RetrievalService<B>, request: HttpRequest) -> HttpResponse {
    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") => html_response(render_home(service, None)),
        ("GET", "/assets/beer.min.css") => static_response("text/css; charset=utf-8", BEER_CSS),
        ("GET", "/assets/beer.min.js") => {
            static_response("application/javascript; charset=utf-8", BEER_JS)
        }
        ("GET", "/assets/material-symbols.css") => {
            static_response("text/css; charset=utf-8", MATERIAL_SYMBOLS_CSS)
        }
        ("GET", "/assets/material-symbols-outlined.ttf") => {
            static_response("font/ttf", MATERIAL_SYMBOLS_FONT)
        }
        _ if request.method == "GET" && request.path.starts_with(EXAMPLE_ASSET_PREFIX) => {
            example_asset_response(&request.path)
        }
        ("GET", "/health") => HttpResponse::ok(
            "application/json; charset=utf-8",
            format!(
                "{{\"pairs\":{},\"candidates\":{}}}",
                service.dataset.len(),
                service.index.entries.len()
            )
            .into_bytes(),
        ),
        ("GET", "/search") => match sample_search_from_query(service, &request.query) {
            Ok((hits, source, top_k)) => {
                html_response(render_results(service, &hits, &source, top_k))
            }
            Err(error) => html_response(render_home(service, Some(&error.to_string()))),
        },
        ("POST", "/search") => match upload_search(service, &request) {
            Ok((hits, source, top_k)) => {
                html_response(render_results(service, &hits, &source, top_k))
            }
            Err(error) => html_response(render_home(service, Some(&error.to_string()))),
        },
        ("POST", "/live/search") if service.live => match live_search(service, &request) {
            Ok(response) => json_response(&response),
            Err(error) => {
                eprintln!("live search failed: {error}");
                error_response(400, &error.to_string())
            }
        },
        ("POST", "/live/search") => error_response(404, "live mode is disabled"),
        _ if request.method == "GET" && request.path.starts_with("/candidate/") => {
            image_response_by_index(&service.index, &request.path, "/candidate/")
        }
        _ if request.method == "GET" && request.path.starts_with("/sample/") => {
            sample_image_response(service, &request.path)
        }
        _ => error_response(404, "not found"),
    }
}

fn example_image_names() -> Vec<String> {
    let Ok(entries) = fs::read_dir(EXAMPLE_ASSET_DIR) else {
        return Vec::new();
    };
    let mut names = entries
        .filter_map(|entry| {
            let entry = entry.ok()?;
            let name = entry.file_name().into_string().ok()?;
            is_example_gallery_image_name(&name).then_some(name)
        })
        .collect::<Vec<_>>();

    names.sort_by(|left, right| {
        example_image_index(left)
            .cmp(&example_image_index(right))
            .then_with(|| left.cmp(right))
    });
    names
}

fn is_example_gallery_image_name(name: &str) -> bool {
    is_example_image_name(name)
        && Path::new(name)
            .extension()
            .and_then(|extension| extension.to_str())
            .map(|extension| {
                matches!(
                    extension.to_ascii_lowercase().as_str(),
                    "avif" | "png" | "jpg" | "jpeg" | "webp"
                )
            })
            .unwrap_or(false)
}

fn example_image_index(name: &str) -> usize {
    Path::new(name)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| stem.parse::<usize>().ok())
        .unwrap_or(usize::MAX)
}

fn is_example_image_name(name: &str) -> bool {
    if name.is_empty() || name == "." || name == ".." || name.contains('/') || name.contains('\\') {
        return false;
    }
    let Some(extension) = Path::new(name)
        .extension()
        .and_then(|extension| extension.to_str())
    else {
        return false;
    };
    matches!(
        extension.to_ascii_lowercase().as_str(),
        "avif" | "png" | "jpg" | "jpeg" | "webp" | "heic" | "heif"
    )
}

fn sample_search_from_query(
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

fn upload_search(
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

fn live_search(
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

fn image_response_by_index(index: &CandidateIndex, path: &str, prefix: &str) -> HttpResponse {
    let entry_index = match path
        .strip_prefix(prefix)
        .and_then(|value| value.parse::<usize>().ok())
    {
        Some(entry_index) => entry_index,
        None => return error_response(400, "invalid image index"),
    };
    let Some(entry) = index.entries.get(entry_index) else {
        return error_response(404, "candidate image not found");
    };
    file_response(&entry.glyph_path)
}

fn sample_image_response(service: &RetrievalService<impl Backend>, path: &str) -> HttpResponse {
    let sample_index = match path
        .strip_prefix("/sample/")
        .and_then(|value| value.parse::<usize>().ok())
    {
        Some(sample_index) => sample_index,
        None => return error_response(400, "invalid sample index"),
    };
    let Some(pair) = service.dataset.pairs().get(sample_index) else {
        return error_response(404, "sample image not found");
    };
    file_response(&pair.image_path)
}

fn example_asset_response(path: &str) -> HttpResponse {
    let Some(name) = path.strip_prefix(EXAMPLE_ASSET_PREFIX) else {
        return error_response(400, "invalid example asset");
    };
    if !is_example_image_name(name) {
        return error_response(400, "invalid example asset");
    }
    file_response(&Path::new(EXAMPLE_ASSET_DIR).join(name))
}

fn file_response(path: &Path) -> HttpResponse {
    match fs::read(path) {
        Ok(bytes) => HttpResponse::ok(image_content_type(path), bytes),
        Err(_) => error_response(404, "image not found"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencoded_parser_decodes_common_form_values() {
        let parsed = parse_urlencoded("sample=12&name=oracle+pose&encoded=%E7%94%B2");

        assert_eq!(parsed.get("sample").map(String::as_str), Some("12"));
        assert_eq!(parsed.get("name").map(String::as_str), Some("oracle pose"));
        assert_eq!(parsed.get("encoded").map(String::as_str), Some("甲"));
    }

    #[test]
    fn multipart_parser_extracts_fields_and_files() {
        let body = concat!(
            "--boundary\r\n",
            "Content-Disposition: form-data; name=\"k\"\r\n\r\n",
            "3\r\n",
            "--boundary\r\n",
            "Content-Disposition: form-data; name=\"image\"; filename=\"pose.png\"\r\n",
            "Content-Type: image/png\r\n\r\n",
            "png-bytes\r\n",
            "--boundary--\r\n"
        );

        let form = parse_multipart(body.as_bytes(), "boundary").expect("multipart form");

        assert_eq!(form.fields.get("k").map(String::as_str), Some("3"));
        assert_eq!(
            form.files.get("image").map(Vec::as_slice),
            Some(&b"png-bytes"[..])
        );
    }

    #[test]
    fn top_k_is_positive_and_capped() {
        assert_eq!(parse_top_k(Some(&"0".to_string()), 8), 8);
        assert_eq!(parse_top_k(Some(&"100".to_string()), 8), 50);
        assert_eq!(parse_top_k(Some(&"7".to_string()), 8), 7);
    }

    #[test]
    fn live_search_response_serializes_candidate_hits() {
        let hit = SearchHit {
            index: 7,
            entry: crate::CandidateEntry {
                id: "jia".to_string(),
                codepoint: Some("U+7532".to_string()),
                character: Some("甲".to_string()),
                persona: "persona_a".to_string(),
                glyph_path: std::path::PathBuf::from("glyph.png"),
                embedding: vec![0.1, 0.2],
            },
            score: 0.875,
        };

        let response = live_search_response(&[hit], 3);
        let http = json_response(&response);
        let body = String::from_utf8(http.body).expect("json utf8");

        assert_eq!(http.status, 200);
        assert_eq!(http.content_type, "application/json; charset=utf-8");
        assert!(body.contains("\"top_k\":3"));
        assert!(body.contains("\"rank\":1"));
        assert!(body.contains("\"image_url\":\"/candidate/7\""));
        assert!(body.contains("\"score\":0.875"));
    }

    #[test]
    fn static_asset_response_uses_embedded_body() {
        let response = static_response("text/css", MATERIAL_SYMBOLS_CSS);

        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "text/css");
        assert!(response.body.starts_with(b"@font-face"));
    }
}
