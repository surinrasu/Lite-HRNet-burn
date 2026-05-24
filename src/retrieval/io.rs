use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Serialize, de::DeserializeOwned};

use super::RetrievalError;

pub(crate) fn canonical_or_original(path: impl AsRef<Path>) -> PathBuf {
    let path = path.as_ref();
    fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

pub(crate) fn recorder_path_stem(path: &Path) -> PathBuf {
    if path
        .extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("mpk"))
    {
        path.with_extension("")
    } else {
        path.to_path_buf()
    }
}

pub(crate) fn write_json_file<T: Serialize>(
    path: impl AsRef<Path>,
    value: &T,
) -> Result<(), RetrievalError> {
    let json = json::to_string_pretty(value)?;
    fs::write(path, json)?;
    Ok(())
}

pub(crate) fn read_json_file<T: DeserializeOwned>(
    path: impl AsRef<Path>,
) -> Result<T, RetrievalError> {
    let mut contents = fs::read(path)?;
    Ok(json::from_slice(&mut contents)?)
}
