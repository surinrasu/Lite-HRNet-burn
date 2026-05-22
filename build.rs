use std::{
    env, fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use burn_onnx::ModelGen;

const OUT_DIR_NAME: &str = "spinepose";

struct ModelSpec {
    filename: &'static str,
    url: &'static str,
}

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        filename: "rfdetr_m_v142_576x576.onnx",
        url: "https://huggingface.co/saifkhichi96/opendetect/resolve/main/rfdetr/rfdetr_m_v142_576x576.onnx",
    },
    ModelSpec {
        filename: "spinepose-l_32xb256-10e_simspine-256x192.onnx",
        url: "https://huggingface.co/dfki-av/spinepose/resolve/main/spinepose-l_32xb256-10e_simspine-256x192.onnx",
    },
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SPINEPOSE_MODEL_DIR");

    let model_paths = MODELS
        .iter()
        .map(|spec| ensure_model(spec).unwrap_or_else(|error| panic!("{error}")))
        .collect::<Vec<_>>();

    let mut generator = ModelGen::new();
    generator.out_dir(OUT_DIR_NAME).development(false);
    for path in &model_paths {
        println!("cargo:rerun-if-changed={}", path.display());
        generator.input(
            path.to_str()
                .unwrap_or_else(|| panic!("model path is not valid UTF-8: {}", path.display())),
        );
    }
    generator.run_from_script();
}

fn ensure_model(spec: &ModelSpec) -> io::Result<PathBuf> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").map_err(io::Error::other)?);
    let model_dir = env::var_os("SPINEPOSE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest_dir.join("assets/models/spinepose"));
    let dst = model_dir.join(spec.filename);
    if dst.is_file() {
        return Ok(dst);
    }

    fs::create_dir_all(&model_dir)?;
    for candidate in local_cache_candidates(&manifest_dir, spec.filename) {
        if candidate.is_file() {
            fs::copy(&candidate, &dst)?;
            return Ok(dst);
        }
    }

    download_model(spec, &dst)?;
    Ok(dst)
}

fn local_cache_candidates(manifest_dir: &Path, filename: &str) -> Vec<PathBuf> {
    let mut candidates = vec![
        manifest_dir
            .join(".spinepose-home/.cache/spinepose/hub/checkpoints")
            .join(filename),
    ];

    if let Some(home) = env::var_os("HOME") {
        candidates.push(
            PathBuf::from(home)
                .join(".cache/spinepose/hub/checkpoints")
                .join(filename),
        );
    }

    candidates
}

fn download_model(spec: &ModelSpec, dst: &Path) -> io::Result<()> {
    let tmp = dst.with_extension("onnx.tmp");
    let status = Command::new("curl")
        .args(["-fL", spec.url, "-o"])
        .arg(&tmp)
        .status()?;

    if !status.success() {
        let _ = fs::remove_file(&tmp);
        return Err(io::Error::other(format!(
            "failed to download {} from {}; set SPINEPOSE_MODEL_DIR or run a pipx spinepose data-generation task once to populate .spinepose-home",
            spec.filename, spec.url
        )));
    }

    fs::rename(tmp, dst)
}
