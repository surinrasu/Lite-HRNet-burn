use ann::{module::Module, prelude::Backend, tensor::Tensor};

use super::layers::{
    ActivationKind, ConditionalChannelWeighting, ConvBnAct, DepthwiseSeparableConv, ShuffleUnit,
    Stem, interpolate_bilinear_aligned, interpolate_nearest,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiteHrModuleType {
    Lite,
    Naive,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HeadUpsampleMode {
    BilinearAligned,
    Nearest,
}

#[derive(Clone, Debug)]
pub struct LiteHrNetConfig {
    pub in_channels: usize,
    pub stem_channels: usize,
    pub stem_out_channels: usize,
    pub stem_expand_ratio: f64,
    pub num_modules: Vec<usize>,
    pub num_branches: Vec<usize>,
    pub num_blocks: Vec<usize>,
    pub module_type: Vec<LiteHrModuleType>,
    pub with_fuse: Vec<bool>,
    pub reduce_ratios: Vec<usize>,
    pub num_channels: Vec<Vec<usize>>,
    pub with_head: bool,
    pub head_upsample_mode: HeadUpsampleMode,
}

impl LiteHrNetConfig {
    pub fn litehrnet18_coco() -> Self {
        Self {
            in_channels: 3,
            stem_channels: 32,
            stem_out_channels: 32,
            stem_expand_ratio: 1.0,
            num_modules: vec![2, 4, 2],
            num_branches: vec![2, 3, 4],
            num_blocks: vec![2, 2, 2],
            module_type: vec![
                LiteHrModuleType::Lite,
                LiteHrModuleType::Lite,
                LiteHrModuleType::Lite,
            ],
            with_fuse: vec![true, true, true],
            reduce_ratios: vec![8, 8, 8],
            num_channels: vec![vec![40, 80], vec![40, 80, 160], vec![40, 80, 160, 320]],
            with_head: true,
            head_upsample_mode: HeadUpsampleMode::BilinearAligned,
        }
    }

    pub fn litehrnet30_coco() -> Self {
        Self {
            num_modules: vec![3, 8, 3],
            ..Self::litehrnet18_coco()
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> LiteHrNet<B> {
        LiteHrNet::new(self.clone(), device)
    }
}

#[derive(Module, Debug)]
pub struct TransitionBranch<B: Backend> {
    pub ops: Vec<ConvBnAct<B>>,
}

impl<B: Backend> TransitionBranch<B> {
    pub fn identity() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn forward(&self, mut input: Tensor<B, 4>) -> Tensor<B, 4> {
        for op in &self.ops {
            input = op.forward(input);
        }
        input
    }
}

#[derive(Module, Debug)]
pub struct TransitionLayer<B: Backend> {
    pub branches: Vec<TransitionBranch<B>>,
}

impl<B: Backend> TransitionLayer<B> {
    pub fn new(
        previous_channels: &[usize],
        current_channels: &[usize],
        device: &B::Device,
    ) -> Self {
        let mut branches = Vec::with_capacity(current_channels.len());

        for (index, current) in current_channels.iter().enumerate() {
            if index < previous_channels.len() {
                if *current == previous_channels[index] {
                    branches.push(TransitionBranch::identity());
                } else {
                    let previous = previous_channels[index];
                    branches.push(TransitionBranch {
                        ops: vec![
                            ConvBnAct::new(
                                previous,
                                previous,
                                3,
                                1,
                                1,
                                previous,
                                true,
                                ActivationKind::None,
                                device,
                            ),
                            ConvBnAct::new(
                                previous,
                                *current,
                                1,
                                1,
                                0,
                                1,
                                true,
                                ActivationKind::Relu,
                                device,
                            ),
                        ],
                    });
                }
            } else {
                let source_channels = *previous_channels.last().expect("previous stage channels");
                let num_downsamples = index + 1 - previous_channels.len();
                let mut ops = Vec::with_capacity(num_downsamples * 2);
                for downsample_index in 0..num_downsamples {
                    let out_channels = if downsample_index + 1 == num_downsamples {
                        *current
                    } else {
                        source_channels
                    };
                    ops.push(ConvBnAct::new(
                        source_channels,
                        source_channels,
                        3,
                        2,
                        1,
                        source_channels,
                        true,
                        ActivationKind::None,
                        device,
                    ));
                    ops.push(ConvBnAct::new(
                        source_channels,
                        out_channels,
                        1,
                        1,
                        0,
                        1,
                        true,
                        ActivationKind::Relu,
                        device,
                    ));
                }
                branches.push(TransitionBranch { ops });
            }
        }

        Self { branches }
    }

    pub fn forward(&self, previous: &[Tensor<B, 4>]) -> Vec<Tensor<B, 4>> {
        self.branches
            .iter()
            .enumerate()
            .map(|(index, branch)| {
                let input = if index >= previous.len() {
                    previous.last().expect("previous branches").clone()
                } else {
                    previous[index].clone()
                };
                branch.forward(input)
            })
            .collect()
    }
}

#[derive(Module, Debug)]
pub struct FuseLayer<B: Backend> {
    pub ops: Vec<ConvBnAct<B>>,
    #[module(skip)]
    pub upsample: bool,
}

impl<B: Backend> FuseLayer<B> {
    pub fn identity() -> Self {
        Self {
            ops: Vec::new(),
            upsample: false,
        }
    }

    pub fn forward(&self, mut input: Tensor<B, 4>, output_size: [usize; 2]) -> Tensor<B, 4> {
        for op in &self.ops {
            input = op.forward(input);
        }

        if self.upsample {
            interpolate_nearest(input, output_size)
        } else {
            input
        }
    }
}

#[derive(Module, Debug)]
pub struct LiteHrModule<B: Backend> {
    pub lite_layers: Vec<ConditionalChannelWeighting<B>>,
    pub naive_layers: Vec<Vec<ShuffleUnit<B>>>,
    pub fuse_layers: Vec<Vec<FuseLayer<B>>>,
    #[module(skip)]
    pub in_channels: Vec<usize>,
    #[module(skip)]
    pub module_type: LiteHrModuleType,
    #[module(skip)]
    pub multiscale_output: bool,
    #[module(skip)]
    pub with_fuse: bool,
}

impl<B: Backend> LiteHrModule<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_branches: usize,
        num_blocks: usize,
        in_channels: Vec<usize>,
        reduce_ratio: usize,
        module_type: LiteHrModuleType,
        multiscale_output: bool,
        with_fuse: bool,
        device: &B::Device,
    ) -> Self {
        assert_eq!(
            num_branches,
            in_channels.len(),
            "num_branches must match in_channels"
        );

        let lite_layers = if module_type == LiteHrModuleType::Lite {
            (0..num_blocks)
                .map(|_| {
                    ConditionalChannelWeighting::new(in_channels.clone(), 1, reduce_ratio, device)
                })
                .collect()
        } else {
            Vec::new()
        };

        let naive_layers = if module_type == LiteHrModuleType::Naive {
            in_channels
                .iter()
                .map(|channels| {
                    (0..num_blocks)
                        .map(|_| ShuffleUnit::new(*channels, *channels, 1, device))
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            Vec::new()
        };

        let fuse_layers = if with_fuse && num_branches > 1 {
            Self::make_fuse_layers(&in_channels, multiscale_output, device)
        } else {
            Vec::new()
        };

        Self {
            lite_layers,
            naive_layers,
            fuse_layers,
            in_channels,
            module_type,
            multiscale_output,
            with_fuse,
        }
    }

    fn make_fuse_layers(
        in_channels: &[usize],
        multiscale_output: bool,
        device: &B::Device,
    ) -> Vec<Vec<FuseLayer<B>>> {
        let num_branches = in_channels.len();
        let num_out_branches = if multiscale_output { num_branches } else { 1 };
        let mut fuse_layers = Vec::with_capacity(num_out_branches);

        for output_index in 0..num_out_branches {
            let mut output_layers = Vec::with_capacity(num_branches);
            for input_index in 0..num_branches {
                if input_index > output_index {
                    output_layers.push(FuseLayer {
                        ops: vec![ConvBnAct::new(
                            in_channels[input_index],
                            in_channels[output_index],
                            1,
                            1,
                            0,
                            1,
                            true,
                            ActivationKind::None,
                            device,
                        )],
                        upsample: true,
                    });
                } else if input_index == output_index {
                    output_layers.push(FuseLayer::identity());
                } else {
                    let mut ops = Vec::new();
                    for downsample_index in 0..(output_index - input_index) {
                        let is_last = downsample_index + 1 == output_index - input_index;
                        let out_channels = if is_last {
                            in_channels[output_index]
                        } else {
                            in_channels[input_index]
                        };
                        ops.push(ConvBnAct::new(
                            in_channels[input_index],
                            in_channels[input_index],
                            3,
                            2,
                            1,
                            in_channels[input_index],
                            true,
                            ActivationKind::None,
                            device,
                        ));
                        ops.push(ConvBnAct::new(
                            in_channels[input_index],
                            out_channels,
                            1,
                            1,
                            0,
                            1,
                            true,
                            if is_last {
                                ActivationKind::None
                            } else {
                                ActivationKind::Relu
                            },
                            device,
                        ));
                    }
                    output_layers.push(FuseLayer {
                        ops,
                        upsample: false,
                    });
                }
            }
            fuse_layers.push(output_layers);
        }

        fuse_layers
    }

    pub fn forward(&self, mut input: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        if self.in_channels.len() == 1 {
            if self.module_type == LiteHrModuleType::Lite {
                return self
                    .lite_layers
                    .iter()
                    .fold(input, |branches, layer| layer.forward(branches));
            }

            let mut branch = input.remove(0);
            for block in &self.naive_layers[0] {
                branch = block.forward(branch);
            }
            return vec![branch];
        }

        let mut output = match self.module_type {
            LiteHrModuleType::Lite => self
                .lite_layers
                .iter()
                .fold(input, |branches, layer| layer.forward(branches)),
            LiteHrModuleType::Naive => input
                .into_iter()
                .zip(self.naive_layers.iter())
                .map(|(mut branch, blocks)| {
                    for block in blocks {
                        branch = block.forward(branch);
                    }
                    branch
                })
                .collect(),
        };

        if self.with_fuse && !self.fuse_layers.is_empty() {
            let mut fused = Vec::with_capacity(self.fuse_layers.len());
            for (output_index, fuse_layer) in self.fuse_layers.iter().enumerate() {
                let [_, _, height, width] = output[output_index].dims();
                let output_size = [height, width];
                let mut y = if output_index == 0 {
                    output[0].clone()
                } else {
                    fuse_layer[0].forward(output[0].clone(), output_size)
                };

                for input_index in 0..self.in_channels.len() {
                    let addend = if output_index == input_index {
                        output[input_index].clone()
                    } else {
                        fuse_layer[input_index].forward(output[input_index].clone(), output_size)
                    };
                    y = y + addend;
                }
                fused.push(ann::tensor::activation::relu(y));
            }
            output = fused;
        } else if !self.multiscale_output {
            output = vec![output.remove(0)];
        }

        output
    }
}

#[derive(Module, Debug)]
pub struct Stage<B: Backend> {
    pub modules: Vec<LiteHrModule<B>>,
}

impl<B: Backend> Stage<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_modules: usize,
        num_branches: usize,
        num_blocks: usize,
        in_channels: Vec<usize>,
        reduce_ratio: usize,
        module_type: LiteHrModuleType,
        with_fuse: bool,
        multiscale_output: bool,
        device: &B::Device,
    ) -> Self {
        let modules = (0..num_modules)
            .map(|module_index| {
                let output_multiscale = multiscale_output || module_index + 1 != num_modules;
                LiteHrModule::new(
                    num_branches,
                    num_blocks,
                    in_channels.clone(),
                    reduce_ratio,
                    module_type,
                    output_multiscale,
                    with_fuse,
                    device,
                )
            })
            .collect();
        Self { modules }
    }

    pub fn forward(&self, input: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        self.modules
            .iter()
            .fold(input, |branches, module| module.forward(branches))
    }
}

#[derive(Module, Debug)]
pub struct IterativeHead<B: Backend> {
    pub projects: Vec<DepthwiseSeparableConv<B>>,
    #[module(skip)]
    pub in_channels: Vec<usize>,
    #[module(skip)]
    pub upsample_mode: HeadUpsampleMode,
}

impl<B: Backend> IterativeHead<B> {
    pub fn new(
        in_channels: Vec<usize>,
        upsample_mode: HeadUpsampleMode,
        device: &B::Device,
    ) -> Self {
        let reversed = in_channels.iter().copied().rev().collect::<Vec<_>>();
        let num_branches = reversed.len();
        let mut projects = Vec::with_capacity(num_branches);

        for index in 0..num_branches {
            let out_channels = if index + 1 == num_branches {
                reversed[index]
            } else {
                reversed[index + 1]
            };
            projects.push(DepthwiseSeparableConv::new(
                reversed[index],
                out_channels,
                3,
                1,
                1,
                ActivationKind::None,
                ActivationKind::Relu,
                device,
            ));
        }

        Self {
            projects,
            in_channels: reversed,
            upsample_mode,
        }
    }

    pub fn forward(&self, input: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        let mut reversed = input.into_iter().rev().collect::<Vec<_>>();
        let mut outputs = Vec::with_capacity(reversed.len());
        let mut last_x: Option<Tensor<B, 4>> = None;

        for (index, mut branch) in reversed.drain(..).enumerate() {
            if let Some(previous) = last_x {
                let [_, _, height, width] = branch.dims();
                let previous = match self.upsample_mode {
                    HeadUpsampleMode::BilinearAligned => {
                        interpolate_bilinear_aligned(previous, [height, width])
                    }
                    HeadUpsampleMode::Nearest => interpolate_nearest(previous, [height, width]),
                };
                branch = branch + previous;
            }

            branch = self.projects[index].forward(branch);
            last_x = Some(branch.clone());
            outputs.push(branch);
        }

        outputs.into_iter().rev().collect()
    }
}

#[derive(Module, Debug)]
pub struct LiteHrNet<B: Backend> {
    pub stem: Stem<B>,
    pub transitions: Vec<TransitionLayer<B>>,
    pub stages: Vec<Stage<B>>,
    pub head_layer: Option<IterativeHead<B>>,
    #[module(skip)]
    pub config: LiteHrNetConfig,
}

impl<B: Backend> LiteHrNet<B> {
    pub fn new(config: LiteHrNetConfig, device: &B::Device) -> Self {
        let stem = Stem::new(
            config.in_channels,
            config.stem_channels,
            config.stem_out_channels,
            config.stem_expand_ratio,
            device,
        );

        let mut transitions = Vec::with_capacity(config.num_channels.len());
        let mut stages = Vec::with_capacity(config.num_channels.len());
        let mut previous_channels = vec![config.stem_out_channels];

        for stage_index in 0..config.num_channels.len() {
            let current_channels = config.num_channels[stage_index].clone();
            transitions.push(TransitionLayer::new(
                &previous_channels,
                &current_channels,
                device,
            ));
            stages.push(Stage::new(
                config.num_modules[stage_index],
                config.num_branches[stage_index],
                config.num_blocks[stage_index],
                current_channels.clone(),
                config.reduce_ratios[stage_index],
                config.module_type[stage_index],
                config.with_fuse[stage_index],
                true,
                device,
            ));
            previous_channels = current_channels;
        }

        let head_layer = config
            .with_head
            .then(|| IterativeHead::new(previous_channels, config.head_upsample_mode, device));

        Self {
            stem,
            transitions,
            stages,
            head_layer,
            config,
        }
    }

    pub fn forward_features(&self, input: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        let stem = self.stem.forward(input);
        let mut branches = vec![stem];

        for (transition, stage) in self.transitions.iter().zip(self.stages.iter()) {
            let stage_input = transition.forward(&branches);
            branches = stage.forward(stage_input);
        }

        if let Some(head_layer) = &self.head_layer {
            head_layer.forward(branches)
        } else {
            branches
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Vec<Tensor<B, 4>> {
        vec![self.forward_features(input).remove(0)]
    }
}

pub const DEFAULT_POSE_JOINTS: usize = 37;

#[derive(Clone, Debug)]
pub struct LiteHrNetPoseConfig {
    pub backbone: LiteHrNetConfig,
    pub num_joints: usize,
}

impl LiteHrNetPoseConfig {
    pub fn litehrnet18_coco() -> Self {
        Self {
            backbone: LiteHrNetConfig::litehrnet18_coco(),
            num_joints: DEFAULT_POSE_JOINTS,
        }
    }

    pub fn litehrnet30_coco() -> Self {
        Self {
            backbone: LiteHrNetConfig::litehrnet30_coco(),
            num_joints: DEFAULT_POSE_JOINTS,
        }
    }

    pub fn init<B: Backend>(&self, device: &B::Device) -> LiteHrNetPose<B> {
        LiteHrNetPose::new(self.clone(), device)
    }
}

#[derive(Module, Debug)]
pub struct LiteHrNetPose<B: Backend> {
    pub backbone: LiteHrNet<B>,
    pub head: ConvBnAct<B>,
    #[module(skip)]
    pub config: LiteHrNetPoseConfig,
}

impl<B: Backend> LiteHrNetPose<B> {
    pub fn new(config: LiteHrNetPoseConfig, device: &B::Device) -> Self {
        let head_in_channels = config.backbone.num_channels.last().expect("stages")[0];
        Self {
            backbone: config.backbone.init(device),
            head: ConvBnAct::new(
                head_in_channels,
                config.num_joints,
                1,
                1,
                0,
                1,
                false,
                ActivationKind::None,
                device,
            ),
            config,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let features = self.backbone.forward(input);
        self.head
            .forward(features.into_iter().next().expect("backbone output"))
    }
}

extern crate ann as burn;
