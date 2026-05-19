use std::{
    fs,
    path::{Path, PathBuf},
};

use burn::backend::{Autodiff, Flex};
use image::{Rgb, RgbImage};
use json::{OwnedValue, json, prelude::*};
use lite_hrnet_burn::{
    CocoPoseDataset, HeadUpsampleMode, LiteHrModuleType, LiteHrNetConfig, LiteHrNetPoseConfig,
    PoseDataConfig, PoseTrainingConfig, train::train_dataset,
};

type AB = Autodiff<Flex>;

fn tiny_backbone_config() -> LiteHrNetConfig {
    LiteHrNetConfig {
        in_channels: 3,
        stem_channels: 16,
        stem_out_channels: 16,
        stem_expand_ratio: 1.0,
        num_modules: vec![1, 1, 1],
        num_branches: vec![2, 3, 4],
        num_blocks: vec![1, 1, 1],
        module_type: vec![
            LiteHrModuleType::Lite,
            LiteHrModuleType::Lite,
            LiteHrModuleType::Lite,
        ],
        with_fuse: vec![true, true, true],
        reduce_ratios: vec![8, 8, 8],
        num_channels: vec![vec![16, 32], vec![16, 32, 64], vec![16, 32, 64, 128]],
        with_head: true,
        head_upsample_mode: HeadUpsampleMode::BilinearAligned,
    }
}

fn fixture_dir(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("lite_hrnet_burn_{name}_{}", std::process::id()))
}

fn write_fixture_dataset(root: &Path) -> (PathBuf, PathBuf) {
    let image_dir = root.join("images");
    fs::create_dir_all(&image_dir).expect("image dir");

    let mut image = RgbImage::new(64, 64);
    for y in 0..64 {
        for x in 0..64 {
            image.put_pixel(x, y, Rgb([x as u8, y as u8, 128]));
        }
    }
    image
        .save(image_dir.join("person.png"))
        .expect("image save");

    let mut keypoints = vec![0.0; 17 * 3];
    keypoints[0] = 32.0;
    keypoints[1] = 32.0;
    keypoints[2] = 2.0;

    let annotation: OwnedValue = json!({
        "images": [{"id": 1, "file_name": "person.png", "width": 64, "height": 64}],
        "annotations": [{
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "bbox": [8.0, 4.0, 40.0, 52.0],
            "area": 2080.0,
            "iscrowd": 0,
            "num_keypoints": 1,
            "keypoints": keypoints
        }],
        "categories": []
    });
    let annotation_path = root.join("annotations.json");
    fs::write(
        &annotation_path,
        json::to_string(&annotation).expect("json"),
    )
    .expect("annotation write");

    (annotation_path, image_dir)
}

#[test]
fn coco_pose_dataset_builds_model_ready_batches() {
    let root = fixture_dir("dataset");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("root");
    let (annotation_path, image_dir) = write_fixture_dataset(&root);

    let data_config = PoseDataConfig::from_input(64, 48, 17);
    let dataset =
        CocoPoseDataset::from_coco(annotation_path, image_dir, data_config).expect("dataset");
    assert_eq!(dataset.len(), 1);

    let device = Default::default();
    let batch = dataset.batch::<AB>(&[0], &device).expect("batch");
    assert_eq!(batch.images.dims(), [1, 3, 64, 48]);
    assert_eq!(batch.targets.dims(), [1, 17, 16, 12]);
    assert_eq!(batch.target_weight.dims(), [1, 17, 1]);

    let weights = batch
        .target_weight
        .into_data()
        .into_vec::<f32>()
        .expect("weights");
    assert_eq!(weights[0], 1.0);
    assert!(weights[1..].iter().all(|value| *value == 0.0));

    let _ = fs::remove_dir_all(root);
}

#[test]
fn dataset_training_loop_writes_report_and_checkpoints() {
    let root = fixture_dir("training");
    let _ = fs::remove_dir_all(&root);
    fs::create_dir_all(&root).expect("root");
    let (annotation_path, image_dir) = write_fixture_dataset(&root);

    let data_config = PoseDataConfig::from_input(64, 48, 17);
    let dataset =
        CocoPoseDataset::from_coco(&annotation_path, &image_dir, data_config).expect("dataset");
    let checkpoint_dir = root.join("checkpoints");
    let config = PoseTrainingConfig {
        model: LiteHrNetPoseConfig {
            backbone: tiny_backbone_config(),
            num_joints: 17,
        },
        epochs: 1,
        batch_size: 1,
        learning_rate: 2e-3,
        shuffle: false,
        seed: 7,
        max_train_samples: None,
        max_val_samples: None,
        checkpoint_dir: checkpoint_dir.clone(),
        log_every: 0,
        save_every_epoch: false,
    };

    let device = Default::default();
    let (_model, report) =
        train_dataset::<AB>(config, dataset.clone(), Some(dataset), &device).expect("train");

    assert_eq!(report.epochs.len(), 1);
    assert!(checkpoint_dir.join("last.mpk").exists());
    assert!(checkpoint_dir.join("best.mpk").exists());
    assert!(checkpoint_dir.join("training_report.json").exists());

    let report_data = fs::read(checkpoint_dir.join("training_report.json")).expect("report");
    let mut report_data = report_data;
    let parsed: OwnedValue = json::from_slice(&mut report_data).expect("report json");
    assert_eq!(parsed["epochs"][0]["epoch"].as_u64(), Some(1));

    let _ = fs::remove_dir_all(root);
}
