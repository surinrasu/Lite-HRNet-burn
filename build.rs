use std::{
    env, fs, io,
    path::{Path, PathBuf},
};

use burn_onnx::ModelGen;

const OUT_DIR_NAME: &str = "spinepose";
const DEFAULT_MODEL_DIR: &str = "assets/models/spinepose";

struct ModelSpec {
    filename: &'static str,
    stem: &'static str,
}

const MODELS: &[ModelSpec] = &[
    ModelSpec {
        filename: "rfdetr_m_v142_576x576.onnx",
        stem: "rfdetr_m_v142_576x576",
    },
    ModelSpec {
        filename: "spinepose-l_32xb256-10e_simspine-256x192.onnx",
        stem: "spinepose-l_32xb256-10e_simspine-256x192",
    },
];

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=SPINEPOSE_MODEL_DIR");
    println!("cargo:rerun-if-env-changed=SPINEPOSE_BURN_DIR");

    if let Some(burn_dir) = env::var_os("SPINEPOSE_BURN_DIR").map(PathBuf::from) {
        copy_preconverted_models(&burn_dir).unwrap_or_else(|error| panic!("{error}"));
        return;
    }

    let model_paths = MODELS
        .iter()
        .map(|spec| resolve_model(spec).unwrap_or_else(|error| panic!("{error}")))
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

fn copy_preconverted_models(burn_dir: &Path) -> io::Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").map_err(io::Error::other)?).join(OUT_DIR_NAME);
    fs::create_dir_all(&out_dir)?;

    for spec in MODELS {
        for extension in ["rs", "bpk"] {
            let source = burn_dir.join(format!("{}.{}", spec.stem, extension));
            if !source.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!(
                        "missing preconverted SpinePose artifact {}; run `mise run models:spinepose:convert` or unset SPINEPOSE_BURN_DIR",
                        source.display()
                    ),
                ));
            }
            println!("cargo:rerun-if-changed={}", source.display());
            fs::copy(&source, out_dir.join(source.file_name().unwrap()))?;
        }
    }

    Ok(())
}

fn resolve_model(spec: &ModelSpec) -> io::Result<PathBuf> {
    for candidate in model_candidates(spec.filename)? {
        if candidate.is_file() {
            return Ok(candidate);
        }
    }

    Err(io::Error::new(
        io::ErrorKind::NotFound,
        format!(
            "missing SpinePose ONNX model {}; run `mise run models:spinepose` or set SPINEPOSE_MODEL_DIR",
            spec.filename
        ),
    ))
}

fn model_candidates(filename: &str) -> io::Result<Vec<PathBuf>> {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").map_err(io::Error::other)?);
    let mut candidates = Vec::new();

    if let Some(model_dir) = env::var_os("SPINEPOSE_MODEL_DIR") {
        candidates.push(PathBuf::from(model_dir).join(filename));
    }

    candidates.push(manifest_dir.join(DEFAULT_MODEL_DIR).join(filename));
    candidates.push(
        manifest_dir
            .join(".spinepose-home/.cache/spinepose/hub/checkpoints")
            .join(filename),
    );

    if let Some(home) = env::var_os("HOME") {
        candidates.push(
            PathBuf::from(home)
                .join(".cache/spinepose/hub/checkpoints")
                .join(filename),
        );
    }

    Ok(candidates)
}
