use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    path::Path,
};

use burn::tensor::backend::Backend;
use hypertext::{Raw, prelude::*};

use crate::{
    CandidateIndex, RetrievalError, RetrievalModel, RetrievalPairDataset, SearchHit,
    encode_pose_features, extract_pose_features_from_bytes, extract_pose_features_from_path,
    search_index,
};

const BEER_CSS: &[u8] = include_bytes!("../assets/retrieval/beer.min.css");
const BEER_JS: &[u8] = include_bytes!("../assets/retrieval/beer.min.js");
const MATERIAL_SYMBOLS_CSS: &[u8] = include_bytes!("../assets/retrieval/material-symbols.css");
const MATERIAL_SYMBOLS_FONT: &[u8] =
    include_bytes!("../assets/retrieval/material-symbols-outlined.ttf");

pub struct RetrievalService<B: Backend> {
    pub model: RetrievalModel<B>,
    pub index: CandidateIndex,
    pub dataset: RetrievalPairDataset,
    pub device: B::Device,
    pub default_top_k: usize,
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
        ("GET", "/assets/retrieval/beer.min.css") => {
            static_response("text/css; charset=utf-8", BEER_CSS)
        }
        ("GET", "/assets/retrieval/beer.min.js") => {
            static_response("application/javascript; charset=utf-8", BEER_JS)
        }
        ("GET", "/assets/retrieval/material-symbols.css") => {
            static_response("text/css; charset=utf-8", MATERIAL_SYMBOLS_CSS)
        }
        ("GET", "/assets/retrieval/material-symbols-outlined.ttf") => {
            static_response("font/ttf", MATERIAL_SYMBOLS_FONT)
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
        _ if request.method == "GET" && request.path.starts_with("/candidate/") => {
            image_response_by_index(&service.index, &request.path, "/candidate/")
        }
        _ if request.method == "GET" && request.path.starts_with("/sample/") => {
            sample_image_response(service, &request.path)
        }
        _ => error_response(404, "not found"),
    }
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
    let features = extract_pose_features_from_path(&pair.image_path)?;
    let embedding = encode_pose_features(&service.model, &features, &service.device)?;
    let hits = search_index(&service.index, &embedding, top_k);
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
        let features = extract_pose_features_from_bytes(image)?;
        let embedding = encode_pose_features(&service.model, &features, &service.device)?;
        let hits = search_index(&service.index, &embedding, top_k);
        return Ok((hits, "uploaded image".to_string(), top_k));
    }

    let body = String::from_utf8_lossy(&request.body);
    let form = parse_urlencoded(&body);
    sample_search_from_query(service, &form)
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

fn file_response(path: &Path) -> HttpResponse {
    match fs::read(path) {
        Ok(bytes) => HttpResponse::ok(image_content_type(path), bytes),
        Err(_) => error_response(404, "image not found"),
    }
}

fn render_home(service: &RetrievalService<impl Backend>, error: Option<&str>) -> String {
    let sample_count = service.dataset.len();
    let candidate_count = service.index.entries.len();
    let top_k = service.default_top_k;
    let sample_indices = (0..sample_count.min(24)).collect::<Vec<_>>();

    maud! {
        !DOCTYPE
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                link rel="stylesheet" href="/assets/retrieval/material-symbols.css";
                link rel="stylesheet" href="/assets/retrieval/beer.min.css";
                script type="module" src="/assets/retrieval/beer.min.js" {}
                style { (Raw::dangerously_create(APP_CSS)) }
                title { "Oracle Pose Retrieval" }
            }
            body.dark {
                main.responsive.max {
                    header class="app-bar" {
                        h5 { "Oracle Pose Retrieval" }
                        div class="stats" {
                            span class="stat-pill" { (sample_count) " pairs" }
                            span class="stat-pill" { (candidate_count) " candidates" }
                        }
                    }

                    @if let Some(error) = error {
                        article.border.error {
                            i { "error" }
                            span { (error) }
                        }
                    }

                    section class="search-panel" {
                        article class="query-card" {
                            h6 { "Query by upload" }
                            form class="query-form" method="post" action="/search" enctype="multipart/form-data" {
                                label class="file-picker" for="query-image" {
                                    i { "upload_file" }
                                    span { "Choose pose image" }
                                }
                                input id="query-image" class="file-input" type="file" name="image" accept="image/png,image/jpeg,image/webp";
                                div class="query-controls" {
                                    div class="topk-control" {
                                        div.field.border.round {
                                            input type="number" name="k" min="1" max="50" value=(top_k);
                                            label { "top k" }
                                        }
                                    }
                                    div.actions {
                                        button.round type="submit" {
                                            i { "search" }
                                            span { "Search" }
                                        }
                                    }
                                }
                            }
                        }
                        article class="query-card" {
                            h6 { "Query by data sample" }
                            form class="query-form" method="get" action="/search" {
                                div class="query-controls sample-controls" {
                                    div class="sample-control" {
                                        div.field.border.round {
                                            input type="number" name="sample" min="0" max=(sample_count.saturating_sub(1)) value="0";
                                            label { "sample id" }
                                        }
                                    }
                                    div class="topk-control" {
                                        div.field.border.round {
                                            input type="number" name="k" min="1" max="50" value=(top_k);
                                            label { "top k" }
                                        }
                                    }
                                    div.actions {
                                        button.round type="submit" {
                                            i { "play_arrow" }
                                            span { "Run" }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    h6 { "Samples" }
                    div class="sample-grid" {
                        @for index in sample_indices.iter().copied() {
                            a class="sample-card" href=(format!("/search?sample={index}&k={top_k}")) {
                                img src=(format!("/sample/{index}")) alt=(service.dataset.pairs()[index].id.as_str()) "loading"="lazy";
                                span { "#" (index) " " (service.dataset.pairs()[index].id.as_str()) }
                            }
                        }
                    }
                }
            }
        }
    }
    .render()
    .as_inner()
    .to_string()
}

fn render_results(
    service: &RetrievalService<impl Backend>,
    hits: &[SearchHit],
    source: &str,
    top_k: usize,
) -> String {
    let sample_count = service.dataset.len();
    let candidate_count = service.index.entries.len();

    maud! {
        !DOCTYPE
        html lang="en" {
            head {
                meta charset="utf-8";
                meta name="viewport" content="width=device-width, initial-scale=1";
                link rel="stylesheet" href="/assets/retrieval/material-symbols.css";
                link rel="stylesheet" href="/assets/retrieval/beer.min.css";
                script type="module" src="/assets/retrieval/beer.min.js" {}
                style { (Raw::dangerously_create(APP_CSS)) }
                title { "Oracle Pose Retrieval" }
            }
            body.dark {
                main.responsive.max {
                    header class="app-bar" {
                        a class="back-link" href="/" aria-label="Back" {
                            i { "arrow_back" }
                        }
                        h5 { "Results" }
                        div class="stats" {
                            span class="stat-pill" { (sample_count) " pairs" }
                            span class="stat-pill" { (candidate_count) " candidates" }
                        }
                    }
                    article class="result-summary" {
                        h6 { (source) }
                        p { "Showing top " (top_k) " candidates by shared embedding similarity." }
                    }
                    div class="result-grid" {
                        @for (rank, hit) in hits.iter().enumerate() {
                            article class="result-card" {
                                div.rank { "#" (rank + 1) }
                                img src=(format!("/candidate/{}", hit.index)) alt=(hit.entry.id.clone()) "loading"="lazy";
                                h6 { (hit.entry.character.clone().unwrap_or_else(|| hit.entry.id.clone())) }
                                p {
                                    (hit.entry.id)
                                    @if let Some(codepoint) = &hit.entry.codepoint {
                                        " " (codepoint)
                                    }
                                }
                                progress max="1" value=(format!("{:.4}", hit.score.max(0.0))) {}
                                small { "score " (format!("{:.4}", hit.score)) }
                            }
                        }
                    }
                }
            }
        }
    }
    .render()
    .as_inner()
    .to_string()
}

const APP_CSS: &str = r#"
main.max { max-width: 1180px; padding-top: 1rem; }
i { font-family: "Material Symbols Outlined"; font-weight: normal; font-style: normal; line-height: 1; letter-spacing: normal; text-transform: none; display: inline-block; white-space: nowrap; }
.app-bar { min-height: 4.25rem; display: flex; align-items: center; gap: 1rem; padding: 0 1rem; margin-bottom: 1rem; border-radius: .5rem; background: var(--surface-container); overflow: hidden; }
.app-bar h5 { margin: 0; line-height: 1.15; }
.back-link { width: 2.75rem; height: 2.75rem; flex: 0 0 2.75rem; display: inline-flex; align-items: center; justify-content: center; border-radius: 50%; color: var(--on-surface); text-decoration: none; }
.back-link:hover { background: var(--surface-container-highest); }
.back-link i { display: block; font-size: 1.9rem; line-height: 1; }
.stats { margin-left: auto; display: flex; flex-wrap: wrap; justify-content: flex-end; gap: .5rem; }
.stat-pill { display: inline-flex; align-items: center; min-height: 1.45rem; padding: 0 .55rem; border-radius: 999px; background: rgb(255 180 171); color: rgb(105 0 5); font-size: .82rem; font-weight: 700; line-height: 1; white-space: nowrap; }
.search-panel { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 1rem; align-items: stretch; }
.query-card { min-height: 13rem; display: flex; flex-direction: column; gap: 1rem; border-radius: .5rem; }
.query-card h6 { margin: 0; }
.query-form { display: flex; flex-direction: column; gap: 1rem; height: 100%; }
.file-picker { min-height: 4.5rem; display: flex; align-items: center; gap: .75rem; padding: 1rem; border: 1px dashed var(--outline); border-radius: .5rem; background: var(--surface-container-high); color: var(--on-surface); cursor: pointer; }
.file-picker i { font-size: 2rem; color: var(--primary); }
.file-picker span { font-weight: 600; overflow-wrap: anywhere; }
.file-input { position: absolute; width: 1px; height: 1px; opacity: 0; pointer-events: none; }
.query-controls { margin-top: auto; display: grid; grid-template-columns: minmax(8rem, 12rem) 1fr; gap: .75rem; align-items: end; }
.sample-controls { grid-template-columns: minmax(10rem, 1fr) minmax(7rem, 10rem) auto; }
.actions { display: flex; align-items: end; justify-content: flex-end; min-height: 4rem; }
.sample-grid, .result-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: .75rem; }
.sample-card { color: inherit; text-decoration: none; border: 1px solid var(--outline-variant); border-radius: .5rem; overflow: hidden; background: var(--surface-container); display: grid; gap: .35rem; padding: .45rem; }
.sample-card img, .result-card img { width: 100%; aspect-ratio: 1 / 1; object-fit: contain; background: white; border-radius: .35rem; }
.sample-card span { font-size: .78rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.result-card { position: relative; border-radius: .5rem; }
.result-card .rank { position: absolute; top: .55rem; left: .55rem; padding: .1rem .45rem; border-radius: 999px; background: var(--primary); color: var(--on-primary); font-size: .75rem; }
.result-card h6, .result-card p { overflow-wrap: anywhere; }
.result-summary { margin-bottom: 1rem; }
@media (max-width: 900px) {
  .search-panel { grid-template-columns: 1fr; }
  .sample-controls, .query-controls { grid-template-columns: 1fr; }
  .actions { justify-content: stretch; }
  .actions button { width: 100%; }
}
@media (max-width: 560px) {
  .app-bar { align-items: flex-start; flex-direction: column; padding: 1rem; }
  .stats { margin-left: 0; justify-content: flex-start; }
}
"#;

#[derive(Debug)]
struct HttpRequest {
    method: String,
    path: String,
    query: BTreeMap<String, String>,
    headers: BTreeMap<String, String>,
    body: Vec<u8>,
}

struct HttpResponse {
    status: u16,
    reason: &'static str,
    content_type: &'static str,
    body: Vec<u8>,
}

impl HttpResponse {
    fn ok(content_type: &'static str, body: Vec<u8>) -> Self {
        Self {
            status: 200,
            reason: "OK",
            content_type,
            body,
        }
    }
}

fn html_response(body: String) -> HttpResponse {
    HttpResponse::ok("text/html; charset=utf-8", body.into_bytes())
}

fn static_response(content_type: &'static str, body: &'static [u8]) -> HttpResponse {
    HttpResponse::ok(content_type, body.to_vec())
}

fn error_response(status: u16, message: &str) -> HttpResponse {
    let reason = match status {
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        _ => "Error",
    };
    HttpResponse {
        status,
        reason,
        content_type: "text/plain; charset=utf-8",
        body: message.as_bytes().to_vec(),
    }
}

fn read_request(stream: &mut TcpStream) -> Result<Option<HttpRequest>, RetrievalError> {
    let mut buffer = Vec::new();
    let mut chunk = [0_u8; 8192];
    let mut header_end = None;

    loop {
        let read = stream.read(&mut chunk)?;
        if read == 0 {
            break;
        }
        buffer.extend_from_slice(&chunk[..read]);
        if header_end.is_none() {
            header_end = find_bytes(&buffer, b"\r\n\r\n").map(|index| index + 4);
        }
        if let Some(header_end) = header_end {
            let content_length = content_length(&buffer[..header_end]);
            if buffer.len() >= header_end + content_length {
                break;
            }
        }
        if buffer.len() > 16 * 1024 * 1024 {
            return Err(RetrievalError::InvalidData("request too large".to_string()));
        }
    }

    if buffer.is_empty() {
        return Ok(None);
    }
    let header_end = header_end.ok_or_else(|| {
        RetrievalError::InvalidData("malformed HTTP request: missing header terminator".to_string())
    })?;
    let header_text = String::from_utf8_lossy(&buffer[..header_end]);
    let mut lines = header_text.lines();
    let request_line = lines
        .next()
        .ok_or_else(|| RetrievalError::InvalidData("missing request line".to_string()))?;
    let mut request_parts = request_line.split_whitespace();
    let method = request_parts.next().unwrap_or("").to_string();
    let target = request_parts.next().unwrap_or("/");
    let (path, query) = split_target(target);

    let mut headers = BTreeMap::new();
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    let content_length = headers
        .get("content-length")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(0);
    let body_end = (header_end + content_length).min(buffer.len());

    Ok(Some(HttpRequest {
        method,
        path,
        query,
        headers,
        body: buffer[header_end..body_end].to_vec(),
    }))
}

fn write_response(stream: &mut TcpStream, response: HttpResponse) -> Result<(), RetrievalError> {
    write!(
        stream,
        "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        response.status,
        response.reason,
        response.content_type,
        response.body.len()
    )?;
    stream.write_all(&response.body)?;
    Ok(())
}

fn content_length(header_bytes: &[u8]) -> usize {
    let headers = String::from_utf8_lossy(header_bytes);
    headers
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0)
}

fn split_target(target: &str) -> (String, BTreeMap<String, String>) {
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    (path.to_string(), parse_urlencoded(query))
}

fn parse_urlencoded(input: &str) -> BTreeMap<String, String> {
    let mut values = BTreeMap::new();
    for part in input.split('&').filter(|part| !part.is_empty()) {
        let (name, value) = part.split_once('=').unwrap_or((part, ""));
        values.insert(percent_decode(name), percent_decode(value));
    }
    values
}

fn percent_decode(input: &str) -> String {
    let mut output = Vec::with_capacity(input.len());
    let bytes = input.as_bytes();
    let mut index = 0;
    while index < bytes.len() {
        match bytes[index] {
            b'+' => {
                output.push(b' ');
                index += 1;
            }
            b'%' if index + 2 < bytes.len() => {
                if let Ok(value) = u8::from_str_radix(&input[index + 1..index + 3], 16) {
                    output.push(value);
                    index += 3;
                } else {
                    output.push(bytes[index]);
                    index += 1;
                }
            }
            value => {
                output.push(value);
                index += 1;
            }
        }
    }
    String::from_utf8_lossy(&output).to_string()
}

#[derive(Default)]
struct MultipartForm {
    fields: BTreeMap<String, String>,
    files: BTreeMap<String, Vec<u8>>,
}

fn multipart_boundary(content_type: &str) -> Option<String> {
    content_type.split(';').find_map(|part| {
        let part = part.trim();
        part.strip_prefix("boundary=")
            .map(|value| value.trim_matches('"').to_string())
    })
}

fn parse_multipart(body: &[u8], boundary: &str) -> Result<MultipartForm, RetrievalError> {
    let delimiter = format!("--{boundary}");
    let delimiter = delimiter.as_bytes();
    let mut form = MultipartForm::default();
    let mut cursor = 0;

    while let Some(start) = find_bytes(&body[cursor..], delimiter) {
        let mut part_start = cursor + start + delimiter.len();
        if body.get(part_start..part_start + 2) == Some(b"--") {
            break;
        }
        if body.get(part_start..part_start + 2) == Some(b"\r\n") {
            part_start += 2;
        }
        let Some(next) = find_bytes(&body[part_start..], delimiter) else {
            break;
        };
        let mut part_end = part_start + next;
        if part_end >= 2 && &body[part_end - 2..part_end] == b"\r\n" {
            part_end -= 2;
        }
        parse_multipart_part(&body[part_start..part_end], &mut form)?;
        cursor = part_start + next;
    }

    Ok(form)
}

fn parse_multipart_part(part: &[u8], form: &mut MultipartForm) -> Result<(), RetrievalError> {
    let Some(header_end) = find_bytes(part, b"\r\n\r\n") else {
        return Ok(());
    };
    let headers = String::from_utf8_lossy(&part[..header_end]);
    let data = &part[header_end + 4..];
    let disposition = headers
        .lines()
        .find(|line| {
            line.to_ascii_lowercase()
                .starts_with("content-disposition:")
        })
        .unwrap_or("");
    let Some(name) = disposition_param(disposition, "name") else {
        return Ok(());
    };
    if disposition_param(disposition, "filename").is_some() {
        form.files.insert(name, data.to_vec());
    } else {
        form.fields
            .insert(name, String::from_utf8_lossy(data).trim().to_string());
    }
    Ok(())
}

fn disposition_param(header: &str, key: &str) -> Option<String> {
    header.split(';').find_map(|part| {
        let (name, value) = part.trim().split_once('=')?;
        (name == key).then(|| value.trim_matches('"').to_string())
    })
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn parse_top_k(value: Option<&String>, default_top_k: usize) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_top_k)
        .min(50)
}

fn image_content_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
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
    fn static_asset_response_uses_embedded_body() {
        let response = static_response("text/css", MATERIAL_SYMBOLS_CSS);

        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "text/css");
        assert!(response.body.starts_with(b"@font-face"));
    }
}
