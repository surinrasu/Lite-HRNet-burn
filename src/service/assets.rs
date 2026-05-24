use std::{fs, path::Path};

use ann::tensor::backend::Backend;

use crate::{CandidateIndex, RetrievalService};

use super::http::{HttpResponse, error_response, image_content_type, static_response};

pub(super) const BEER_CSS: &[u8] = include_bytes!("../../assets/beer.min.css");
pub(super) const BEER_JS: &[u8] = include_bytes!("../../assets/beer.min.js");
pub(super) const MATERIAL_SYMBOLS_CSS: &[u8] = include_bytes!("../../assets/material-symbols.css");
pub(super) const MATERIAL_SYMBOLS_FONT: &[u8] =
    include_bytes!("../../assets/material-symbols-outlined.ttf");

const EXAMPLE_ASSET_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/assets/examples");
pub(super) const EXAMPLE_ASSET_PREFIX: &str = "/assets/examples/";

pub(super) fn static_asset_response(path: &str) -> Option<HttpResponse> {
    match path {
        "/assets/beer.min.css" => Some(static_response("text/css; charset=utf-8", BEER_CSS)),
        "/assets/beer.min.js" => Some(static_response(
            "application/javascript; charset=utf-8",
            BEER_JS,
        )),
        "/assets/material-symbols.css" => Some(static_response(
            "text/css; charset=utf-8",
            MATERIAL_SYMBOLS_CSS,
        )),
        "/assets/material-symbols-outlined.ttf" => {
            Some(static_response("font/ttf", MATERIAL_SYMBOLS_FONT))
        }
        _ if path.starts_with(EXAMPLE_ASSET_PREFIX) => Some(example_asset_response(path)),
        _ => None,
    }
}

pub(super) fn example_image_names() -> Vec<String> {
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

pub(super) fn image_response_by_index(
    index: &CandidateIndex,
    path: &str,
    prefix: &str,
) -> HttpResponse {
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

pub(super) fn sample_image_response(
    service: &RetrievalService<impl Backend>,
    path: &str,
) -> HttpResponse {
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
