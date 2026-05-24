use std::net::{TcpListener, TcpStream};

use ann::tensor::backend::Backend;

use crate::{
    CandidateIndex, DefaultPoseEstimator, RetrievalError, RetrievalModel, RetrievalPairDataset,
};

mod assets;
mod http;
mod live;
mod search;
mod views;

use self::{
    assets::{image_response_by_index, sample_image_response, static_asset_response},
    http::{
        HttpRequest, HttpResponse, error_response, html_response, json_response, read_request,
        write_response,
    },
    search::{live_search, sample_search_from_query, upload_search},
    views::{render_home, render_results},
};

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
    if request.method == "GET"
        && let Some(response) = static_asset_response(&request.path)
    {
        return response;
    }

    match (request.method.as_str(), request.path.as_str()) {
        ("GET", "/") => html_response(render_home(service, None)),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlencoded_parser_decodes_common_form_values() {
        let parsed = http::parse_urlencoded("sample=12&name=oracle+pose&encoded=%E7%94%B2");

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

        let form = http::parse_multipart(body.as_bytes(), "boundary").expect("multipart form");

        assert_eq!(form.fields.get("k").map(String::as_str), Some("3"));
        assert_eq!(
            form.files.get("image").map(Vec::as_slice),
            Some(&b"png-bytes"[..])
        );
    }

    #[test]
    fn top_k_is_positive_and_capped() {
        assert_eq!(http::parse_top_k(Some(&"0".to_string()), 8), 8);
        assert_eq!(http::parse_top_k(Some(&"100".to_string()), 8), 50);
        assert_eq!(http::parse_top_k(Some(&"7".to_string()), 8), 7);
    }

    #[test]
    fn live_search_response_serializes_candidate_hits() {
        let hit = crate::SearchHit {
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

        let response = live::live_search_response(&[hit], 3);
        let http = http::json_response(&response);
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
        let response = http::static_response("text/css", assets::MATERIAL_SYMBOLS_CSS);

        assert_eq!(response.status, 200);
        assert_eq!(response.content_type, "text/css");
        assert!(response.body.starts_with(b"@font-face"));
    }
}
