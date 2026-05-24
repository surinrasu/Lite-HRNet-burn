use std::{
    error::Error,
    fmt::{Display, Formatter},
};

#[derive(Debug)]
pub enum PoseDataError {
    Io(std::io::Error),
    Json(json::Error),
    Image(image::ImageError),
    InvalidDataset(String),
}

impl Display for PoseDataError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(error) => write!(formatter, "io error: {error}"),
            Self::Json(error) => write!(formatter, "json error: {error}"),
            Self::Image(error) => write!(formatter, "image error: {error}"),
            Self::InvalidDataset(message) => write!(formatter, "invalid dataset: {message}"),
        }
    }
}

impl Error for PoseDataError {}

impl From<std::io::Error> for PoseDataError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<json::Error> for PoseDataError {
    fn from(value: json::Error) -> Self {
        Self::Json(value)
    }
}

impl From<image::ImageError> for PoseDataError {
    fn from(value: image::ImageError) -> Self {
        Self::Image(value)
    }
}
