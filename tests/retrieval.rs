use std::{
    fs,
    path::{Path, PathBuf},
};

use ann::backend::{Autodiff, Flex};
use image::{Rgb, RgbImage};
use pose_obc_retrieval::{
    RetrievalModelConfig, RetrievalPairDataset, RetrievalTrainingConfig, build_candidate_index,
    extract_glyph_features_from_path, extract_pose_features_from_path, read_candidate_index,
    search_index, train_retrieval_dataset, write_candidate_index,
};

type AB = Autodiff<Flex>;

fn fixture_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "pose_obc_retrieval_retrieval_{name}_{}",
        std::process::id()
    ))
}

fn write_fixture_image(path: &Path, variant: u8) {
    let mut image = RgbImage::from_pixel(32, 32, Rgb([255, 255, 255]));
    for y in 6..26 {
        for x in 6..26 {
            let draw = if variant == 0 {
                (14..=18).contains(&x)
            } else {
                (14..=18).contains(&y)
            };
            if draw {
                image.put_pixel(x, y, Rgb([20, 20, 20]));
            }
        }
    }
    image.save(path).expect("fixture image save");
}

fn write_fixture_dataset(root: &Path) -> PathBuf {
    let data_root = root.join("data");
    let image_dir = data_root.join("persona_fixture").join("images");
    let glyph_dir = data_root.join("persona_fixture").join("glyphs");
    fs::create_dir_all(&image_dir).expect("image dir");
    fs::create_dir_all(&glyph_dir).expect("glyph dir");

    for (name, variant) in [("U+4E00.png", 0), ("U+4E8C.png", 1)] {
        write_fixture_image(&image_dir.join(name), variant);
        write_fixture_image(&glyph_dir.join(name), variant);
    }

    data_root
}

#[test]
fn retrieval_training_index_and_search_workflow_runs() {
    let root = fixture_dir("workflow");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("root");
    let data_root = write_fixture_dataset(&root);

    let dataset = RetrievalPairDataset::from_data_root(&data_root).expect("dataset");
    assert_eq!(dataset.len(), 2);

    let pose_features =
        extract_pose_features_from_path(&dataset.pairs()[0].image_path).expect("pose features");
    let glyph_features =
        extract_glyph_features_from_path(&dataset.pairs()[0].glyph_path).expect("glyph features");
    assert_eq!(
        pose_features.len(),
        pose_obc_retrieval::RETRIEVAL_FEATURE_DIM
    );
    assert_eq!(
        glyph_features.len(),
        pose_obc_retrieval::RETRIEVAL_FEATURE_DIM
    );

    let checkpoint_dir = root.join("checkpoints");
    let config = RetrievalTrainingConfig {
        model: RetrievalModelConfig {
            input_dim: pose_obc_retrieval::RETRIEVAL_FEATURE_DIM,
            hidden_dim: 8,
            embedding_dim: 4,
        },
        epochs: 1,
        batch_size: 2,
        learning_rate: 1e-3,
        temperature: 0.07,
        shuffle: false,
        seed: 7,
        max_pairs: None,
        checkpoint_dir,
        log_every: 0,
        save_every_epoch: false,
    };

    let device = Default::default();
    let (model, report) =
        train_retrieval_dataset::<AB, _>(config.clone(), &dataset, &device, |_| {})
            .expect("train retrieval");
    assert_eq!(report.epochs.len(), 1);

    let index = build_candidate_index(&model, config.model.clone(), &dataset, true, &device)
        .expect("candidate index");
    assert_eq!(index.entries.len(), 2);

    let index_path = root.join("glyph_index.json");
    write_candidate_index(&index_path, &index).expect("write index");
    let index = read_candidate_index(&index_path).expect("read index");
    let query = pose_obc_retrieval::encode_pose_features(&model, &pose_features, &device)
        .expect("query embedding");
    let hits = search_index(&index, &query, 1);
    assert_eq!(hits.len(), 1);

    let _ = fs::remove_dir_all(root);
}
