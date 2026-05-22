use std::{
    collections::BTreeMap,
    io::{Read, Write},
    net::TcpStream,
    path::Path,
};

use serde::Serialize;

use crate::RetrievalError;

#[derive(Debug)]
pub(super) struct HttpRequest {
    pub(super) method: String,
    pub(super) path: String,
    pub(super) query: BTreeMap<String, String>,
    pub(super) headers: BTreeMap<String, String>,
    pub(super) body: Vec<u8>,
}

#[derive(Debug)]
pub(super) struct HttpResponse {
    pub(super) status: u16,
    reason: &'static str,
    pub(super) content_type: &'static str,
    pub(super) body: Vec<u8>,
}

impl HttpResponse {
    pub(super) fn ok(content_type: &'static str, body: Vec<u8>) -> Self {
        Self {
            status: 200,
            reason: "OK",
            content_type,
            body,
        }
    }
}

pub(super) fn html_response(body: String) -> HttpResponse {
    HttpResponse::ok("text/html; charset=utf-8", body.into_bytes())
}

pub(super) fn json_response<T: Serialize>(value: &T) -> HttpResponse {
    match json::to_string(value) {
        Ok(body) => HttpResponse::ok("application/json; charset=utf-8", body.into_bytes()),
        Err(error) => error_response(500, &format!("failed to serialize response: {error}")),
    }
}

pub(super) fn static_response(content_type: &'static str, body: &'static [u8]) -> HttpResponse {
    HttpResponse::ok(content_type, body.to_vec())
}

pub(super) fn error_response(status: u16, message: &str) -> HttpResponse {
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

pub(super) fn read_request(stream: &mut TcpStream) -> Result<Option<HttpRequest>, RetrievalError> {
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

pub(super) fn write_response(
    stream: &mut TcpStream,
    response: HttpResponse,
) -> Result<(), RetrievalError> {
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

pub(super) fn parse_urlencoded(input: &str) -> BTreeMap<String, String> {
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
pub(super) struct MultipartForm {
    pub(super) fields: BTreeMap<String, String>,
    pub(super) files: BTreeMap<String, Vec<u8>>,
}

pub(super) fn multipart_boundary(content_type: &str) -> Option<String> {
    content_type.split(';').find_map(|part| {
        let part = part.trim();
        part.strip_prefix("boundary=")
            .map(|value| value.trim_matches('"').to_string())
    })
}

pub(super) fn parse_multipart(
    body: &[u8],
    boundary: &str,
) -> Result<MultipartForm, RetrievalError> {
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

pub(super) fn parse_top_k(value: Option<&String>, default_top_k: usize) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default_top_k)
        .min(50)
}

pub(super) fn image_content_type(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|extension| extension.to_str())
        .unwrap_or("")
        .to_ascii_lowercase()
        .as_str()
    {
        "avif" => "image/avif",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "heic" => "image/heic",
        "heif" => "image/heif",
        _ => "image/png",
    }
}
