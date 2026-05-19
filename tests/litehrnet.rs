use ann::{
    backend::{Autodiff, Flex},
    tensor::{Distribution, Tensor, TensorData},
};
use pose_obc_retrieval::{
    ConditionalChannelWeighting, CrossResolutionWeighting, HeadUpsampleMode, IterativeHead,
    LiteHrModule, LiteHrModuleType, LiteHrNetConfig, LiteHrNetPoseConfig, ShuffleUnit,
    SpatialWeighting, Stem, channel_shuffle,
    train::{run_synthetic_training, synthetic_pose_batch, train_step},
};

type B = Flex;
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

#[test]
fn channel_shuffle_matches_reference_order() {
    let device = Default::default();
    let input = Tensor::<B, 4>::from_data(
        TensorData::new(
            vec![0.0_f32, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            [1, 4, 1, 2],
        ),
        &device,
    );

    let output = channel_shuffle(input, 2);
    let expected = TensorData::new(
        vec![0.0_f32, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0],
        [1, 4, 1, 2],
    );

    output.to_data().assert_eq(&expected, false);
}

#[test]
fn stem_matches_reference_shape_contract() {
    let device = Default::default();
    let stem = Stem::<B>::new(3, 32, 32, 1.0, &device);
    let input = Tensor::random([2, 3, 128, 96], Distribution::Default, &device);

    let output = stem.forward(input);

    assert_eq!(output.dims(), [2, 32, 32, 24]);
}

#[test]
fn weighting_blocks_preserve_branch_shapes() {
    let device = Default::default();
    let spatial = SpatialWeighting::<B>::new(20, 4, &device);
    let spatial_input = Tensor::random([2, 20, 16, 12], Distribution::Default, &device);
    assert_eq!(spatial.forward(spatial_input).dims(), [2, 20, 16, 12]);

    let cross = CrossResolutionWeighting::<B>::new(vec![20, 40], 8, &device);
    let cross_outputs = cross.forward(vec![
        Tensor::random([2, 20, 16, 12], Distribution::Default, &device),
        Tensor::random([2, 40, 8, 6], Distribution::Default, &device),
    ]);
    assert_eq!(cross_outputs[0].dims(), [2, 20, 16, 12]);
    assert_eq!(cross_outputs[1].dims(), [2, 40, 8, 6]);

    let conditional = ConditionalChannelWeighting::<B>::new(vec![40, 80], 1, 8, &device);
    let conditional_outputs = conditional.forward(vec![
        Tensor::random([2, 40, 16, 12], Distribution::Default, &device),
        Tensor::random([2, 80, 8, 6], Distribution::Default, &device),
    ]);
    assert_eq!(conditional_outputs[0].dims(), [2, 40, 16, 12]);
    assert_eq!(conditional_outputs[1].dims(), [2, 80, 8, 6]);
}

#[test]
fn shuffle_unit_and_lite_module_match_branch_shape_contracts() {
    let device = Default::default();
    let shuffle = ShuffleUnit::<B>::new(40, 40, 1, &device);
    let shuffle_input = Tensor::random([2, 40, 16, 12], Distribution::Default, &device);
    assert_eq!(shuffle.forward(shuffle_input).dims(), [2, 40, 16, 12]);

    let module = LiteHrModule::<B>::new(
        2,
        1,
        vec![40, 80],
        8,
        LiteHrModuleType::Lite,
        true,
        true,
        &device,
    );
    let outputs = module.forward(vec![
        Tensor::random([2, 40, 16, 12], Distribution::Default, &device),
        Tensor::random([2, 80, 8, 6], Distribution::Default, &device),
    ]);

    assert_eq!(outputs[0].dims(), [2, 40, 16, 12]);
    assert_eq!(outputs[1].dims(), [2, 80, 8, 6]);
}

#[test]
fn iterative_head_progressively_projects_low_to_high_resolution() {
    let device = Default::default();
    let head = IterativeHead::<B>::new(
        vec![40, 80, 160, 320],
        HeadUpsampleMode::BilinearAligned,
        &device,
    );

    let outputs = head.forward(vec![
        Tensor::random([2, 40, 16, 12], Distribution::Default, &device),
        Tensor::random([2, 80, 8, 6], Distribution::Default, &device),
        Tensor::random([2, 160, 4, 3], Distribution::Default, &device),
        Tensor::random([2, 320, 2, 2], Distribution::Default, &device),
    ]);

    assert_eq!(outputs[0].dims(), [2, 40, 16, 12]);
    assert_eq!(outputs[1].dims(), [2, 40, 8, 6]);
    assert_eq!(outputs[2].dims(), [2, 80, 4, 3]);
    assert_eq!(outputs[3].dims(), [2, 160, 2, 2]);
}

#[test]
fn litehrnet18_backbone_and_pose_head_match_coco_shapes() {
    let device = Default::default();
    let backbone = LiteHrNetConfig::litehrnet18_coco().init::<B>(&device);
    let input = Tensor::random([1, 3, 64, 48], Distribution::Default, &device);

    let outputs = backbone.forward(input);

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].dims(), [1, 40, 16, 12]);

    let pose = LiteHrNetPoseConfig::litehrnet18_coco().init::<B>(&device);
    let input = Tensor::random([1, 3, 64, 48], Distribution::Default, &device);
    let heatmaps = pose.forward(input);
    assert_eq!(heatmaps.dims(), [1, 17, 16, 12]);
}

#[test]
fn synthetic_training_closure_runs_forward_backward_and_adam() {
    let device = Default::default();
    let config = LiteHrNetPoseConfig {
        backbone: tiny_backbone_config(),
        num_joints: 17,
    };

    let model = config.init::<AB>(&device);
    let mut optimizer = ann::optim::AdamConfig::new().init::<AB, _>();
    let batch = synthetic_pose_batch::<AB>(1, 64, 48, 17, &device);
    let _model = train_step(model, &mut optimizer, batch, 2e-3);
}

#[test]
fn synthetic_training_loop_returns_updated_model() {
    let device = Default::default();
    let config = LiteHrNetPoseConfig {
        backbone: tiny_backbone_config(),
        num_joints: 17,
    };

    let _model = run_synthetic_training::<AB>(config, &device, 1, 1, 64, 48, 2e-3);
}
