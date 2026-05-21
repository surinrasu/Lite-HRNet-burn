use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug)]
pub enum RetrievalError {
    Io(std::io::Error),
    Image(image::ImageError),
    Json(json::Error),
    Recorder(ann::record::RecorderError),
    InvalidData(String),
    Tensor(String),
}

impl Display for RetrievalError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "io error: {error}"),
            Self::Image(error) => write!(formatter, "image error: {error}"),
            Self::Json(error) => write!(formatter, "json error: {error}"),
            Self::Recorder(error) => write!(formatter, "recorder error: {error}"),
            Self::InvalidData(message) => write!(formatter, "invalid data: {message}"),
            Self::Tensor(message) => write!(formatter, "tensor error: {message}"),
        }
    }
}

impl Error for RetrievalError {}

impl From<std::io::Error> for RetrievalError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<image::ImageError> for RetrievalError {
    fn from(value: image::ImageError) -> Self {
        Self::Image(value)
    }
}

impl From<json::Error> for RetrievalError {
    fn from(value: json::Error) -> Self {
        Self::Json(value)
    }
}
