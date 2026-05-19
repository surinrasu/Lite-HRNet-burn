use burn::{
    module::{Initializer, Module},
    nn::{
        BatchNorm, BatchNormConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2dConfig, InterpolateMode},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::Backend,
    tensor::{Tensor, activation, module::adaptive_avg_pool2d},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActivationKind {
    None,
    Relu,
    Sigmoid,
}

impl ActivationKind {
    fn forward<B: Backend>(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match self {
            Self::None => input,
            Self::Relu => activation::relu(input),
            Self::Sigmoid => activation::sigmoid(input),
        }
    }
}

#[derive(Module, Debug)]
pub struct ConvBnAct<B: Backend> {
    pub conv: Conv2d<B>,
    pub bn: Option<BatchNorm<B>>,
    #[module(skip)]
    pub act: ActivationKind,
}

impl<B: Backend> ConvBnAct<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        with_bn: bool,
        act: ActivationKind,
        device: &B::Device,
    ) -> Self {
        let bias = !with_bn;
        let mut conv = Conv2dConfig::new([in_channels, out_channels], [kernel_size, kernel_size])
            .with_stride([stride, stride])
            .with_padding(PaddingConfig2d::Explicit(
                padding, padding, padding, padding,
            ))
            .with_groups(groups)
            .with_bias(bias)
            .with_initializer(Initializer::Normal {
                mean: 0.0,
                std: 0.001,
            })
            .init(device);
        if bias {
            conv.bias = Some(Initializer::Zeros.init([out_channels], device));
        }
        let bn = with_bn.then(|| BatchNormConfig::new(out_channels).init(device));

        Self { conv, bn, act }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.conv.forward(input);
        let output = match &self.bn {
            Some(bn) => bn.forward(output),
            None => output,
        };
        self.act.forward(output)
    }
}

#[derive(Module, Debug)]
pub struct DepthwiseSeparableConv<B: Backend> {
    pub depthwise: ConvBnAct<B>,
    pub pointwise: ConvBnAct<B>,
}

impl<B: Backend> DepthwiseSeparableConv<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        depthwise_act: ActivationKind,
        pointwise_act: ActivationKind,
        device: &B::Device,
    ) -> Self {
        Self {
            depthwise: ConvBnAct::new(
                in_channels,
                in_channels,
                kernel_size,
                stride,
                padding,
                in_channels,
                true,
                depthwise_act,
                device,
            ),
            pointwise: ConvBnAct::new(
                in_channels,
                out_channels,
                1,
                1,
                0,
                1,
                true,
                pointwise_act,
                device,
            ),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        self.pointwise.forward(self.depthwise.forward(input))
    }
}

pub fn channel_shuffle<B: Backend>(input: Tensor<B, 4>, groups: usize) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.dims();
    assert_eq!(channels % groups, 0, "channels must be divisible by groups");

    input
        .reshape([batch, groups, channels / groups, height, width])
        .swap_dims(1, 2)
        .reshape([batch, channels, height, width])
}

#[derive(Module, Debug)]
pub struct SpatialWeighting<B: Backend> {
    pub global_avgpool: AdaptiveAvgPool2d,
    pub conv1: ConvBnAct<B>,
    pub conv2: ConvBnAct<B>,
}

impl<B: Backend> SpatialWeighting<B> {
    pub fn new(channels: usize, ratio: usize, device: &B::Device) -> Self {
        let mid_channels = usize::max(1, channels / ratio);
        Self {
            global_avgpool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            conv1: ConvBnAct::new(
                channels,
                mid_channels,
                1,
                1,
                0,
                1,
                false,
                ActivationKind::Relu,
                device,
            ),
            conv2: ConvBnAct::new(
                mid_channels,
                channels,
                1,
                1,
                0,
                1,
                false,
                ActivationKind::Sigmoid,
                device,
            ),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let weights = self.conv2.forward(
            self.conv1
                .forward(self.global_avgpool.forward(input.clone())),
        );
        input * weights
    }
}

#[derive(Module, Debug)]
pub struct CrossResolutionWeighting<B: Backend> {
    pub conv1: ConvBnAct<B>,
    pub conv2: ConvBnAct<B>,
    #[module(skip)]
    pub channels: Vec<usize>,
}

impl<B: Backend> CrossResolutionWeighting<B> {
    pub fn new(channels: Vec<usize>, ratio: usize, device: &B::Device) -> Self {
        let total_channels = channels.iter().sum::<usize>();
        let mid_channels = usize::max(1, total_channels / ratio);

        Self {
            conv1: ConvBnAct::new(
                total_channels,
                mid_channels,
                1,
                1,
                0,
                1,
                true,
                ActivationKind::Relu,
                device,
            ),
            conv2: ConvBnAct::new(
                mid_channels,
                total_channels,
                1,
                1,
                0,
                1,
                true,
                ActivationKind::Sigmoid,
                device,
            ),
            channels,
        }
    }

    pub fn forward(&self, input: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        let mini_size = {
            let dims = input.last().expect("at least one branch").dims();
            [dims[2], dims[3]]
        };

        let mut pooled = Vec::with_capacity(input.len());
        for (index, branch) in input.iter().enumerate() {
            if index + 1 == input.len() {
                pooled.push(branch.clone());
            } else {
                pooled.push(adaptive_avg_pool2d(branch.clone(), mini_size));
            }
        }

        let weights = self
            .conv2
            .forward(self.conv1.forward(Tensor::cat(pooled, 1)));
        let mut splits = Vec::with_capacity(self.channels.len());
        let mut start = 0;
        for channels in &self.channels {
            splits.push(weights.clone().narrow(1, start, *channels));
            start += channels;
        }

        input
            .into_iter()
            .zip(splits)
            .map(|(branch, weights)| {
                let [_, _, height, width] = branch.dims();
                let weights = interpolate_nearest(weights, [height, width]);
                branch * weights
            })
            .collect()
    }
}

#[derive(Module, Debug)]
pub struct ConditionalChannelWeighting<B: Backend> {
    pub cross_resolution_weighting: CrossResolutionWeighting<B>,
    pub depthwise_convs: Vec<ConvBnAct<B>>,
    pub spatial_weighting: Vec<SpatialWeighting<B>>,
}

impl<B: Backend> ConditionalChannelWeighting<B> {
    pub fn new(
        in_channels: Vec<usize>,
        stride: usize,
        reduce_ratio: usize,
        device: &B::Device,
    ) -> Self {
        assert!(stride == 1 || stride == 2, "stride must be 1 or 2");
        let branch_channels = in_channels
            .iter()
            .map(|channel| {
                assert_eq!(channel % 2, 0, "branch channel count must be even");
                channel / 2
            })
            .collect::<Vec<_>>();

        let depthwise_convs = branch_channels
            .iter()
            .map(|channel| {
                ConvBnAct::new(
                    *channel,
                    *channel,
                    3,
                    stride,
                    1,
                    *channel,
                    true,
                    ActivationKind::None,
                    device,
                )
            })
            .collect();
        let spatial_weighting = branch_channels
            .iter()
            .map(|channel| SpatialWeighting::new(*channel, 4, device))
            .collect();

        Self {
            cross_resolution_weighting: CrossResolutionWeighting::new(
                branch_channels,
                reduce_ratio,
                device,
            ),
            depthwise_convs,
            spatial_weighting,
        }
    }

    pub fn forward(&self, input: Vec<Tensor<B, 4>>) -> Vec<Tensor<B, 4>> {
        let mut x1 = Vec::with_capacity(input.len());
        let mut x2 = Vec::with_capacity(input.len());

        for branch in input {
            let chunks = branch.chunk(2, 1);
            x1.push(chunks[0].clone());
            x2.push(chunks[1].clone());
        }

        let x2 = self.cross_resolution_weighting.forward(x2);
        let x2 = x2
            .into_iter()
            .zip(self.depthwise_convs.iter())
            .map(|(branch, depthwise)| depthwise.forward(branch))
            .collect::<Vec<_>>();
        let x2 = x2
            .into_iter()
            .zip(self.spatial_weighting.iter())
            .map(|(branch, weighting)| weighting.forward(branch))
            .collect::<Vec<_>>();

        x1.into_iter()
            .zip(x2)
            .map(|(branch1, branch2)| channel_shuffle(Tensor::cat(vec![branch1, branch2], 1), 2))
            .collect()
    }
}

#[derive(Module, Debug)]
pub struct Stem<B: Backend> {
    pub conv1: ConvBnAct<B>,
    pub branch1_depthwise: ConvBnAct<B>,
    pub branch1_pointwise: ConvBnAct<B>,
    pub expand_conv: ConvBnAct<B>,
    pub depthwise_conv: ConvBnAct<B>,
    pub linear_conv: ConvBnAct<B>,
    #[module(skip)]
    pub out_channels: usize,
}

impl<B: Backend> Stem<B> {
    pub fn new(
        in_channels: usize,
        stem_channels: usize,
        out_channels: usize,
        expand_ratio: f64,
        device: &B::Device,
    ) -> Self {
        let mid_channels = (stem_channels as f64 * expand_ratio).round() as usize;
        let branch_channels = stem_channels / 2;
        let inc_channels = if stem_channels == out_channels {
            out_channels - branch_channels
        } else {
            out_channels - stem_channels
        };
        let linear_out_channels = if stem_channels == out_channels {
            branch_channels
        } else {
            stem_channels
        };

        Self {
            conv1: ConvBnAct::new(
                in_channels,
                stem_channels,
                3,
                2,
                1,
                1,
                true,
                ActivationKind::Relu,
                device,
            ),
            branch1_depthwise: ConvBnAct::new(
                branch_channels,
                branch_channels,
                3,
                2,
                1,
                branch_channels,
                true,
                ActivationKind::None,
                device,
            ),
            branch1_pointwise: ConvBnAct::new(
                branch_channels,
                inc_channels,
                1,
                1,
                0,
                1,
                true,
                ActivationKind::Relu,
                device,
            ),
            expand_conv: ConvBnAct::new(
                branch_channels,
                mid_channels,
                1,
                1,
                0,
                1,
                true,
                ActivationKind::Relu,
                device,
            ),
            depthwise_conv: ConvBnAct::new(
                mid_channels,
                mid_channels,
                3,
                2,
                1,
                mid_channels,
                true,
                ActivationKind::None,
                device,
            ),
            linear_conv: ConvBnAct::new(
                mid_channels,
                linear_out_channels,
                1,
                1,
                0,
                1,
                true,
                ActivationKind::Relu,
                device,
            ),
            out_channels,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let input = self.conv1.forward(input);
        let chunks = input.chunk(2, 1);
        let x1 = chunks[0].clone();
        let x2 = chunks[1].clone();

        let x1 = self
            .branch1_pointwise
            .forward(self.branch1_depthwise.forward(x1));
        let x2 = self
            .linear_conv
            .forward(self.depthwise_conv.forward(self.expand_conv.forward(x2)));

        channel_shuffle(Tensor::cat(vec![x1, x2], 1), 2)
    }
}

#[derive(Module, Debug)]
pub struct ShuffleUnit<B: Backend> {
    pub branch1_depthwise: Option<ConvBnAct<B>>,
    pub branch1_pointwise: Option<ConvBnAct<B>>,
    pub branch2_expand: ConvBnAct<B>,
    pub branch2_depthwise: ConvBnAct<B>,
    pub branch2_linear: ConvBnAct<B>,
    #[module(skip)]
    pub stride: usize,
}

impl<B: Backend> ShuffleUnit<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &B::Device) -> Self {
        let branch_features = out_channels / 2;
        if stride == 1 {
            assert_eq!(
                in_channels,
                branch_features * 2,
                "stride=1 requires in_channels == out_channels"
            );
        }

        let branch1_depthwise = (stride > 1).then(|| {
            ConvBnAct::new(
                in_channels,
                in_channels,
                3,
                stride,
                1,
                in_channels,
                true,
                ActivationKind::None,
                device,
            )
        });
        let branch1_pointwise = (stride > 1).then(|| {
            ConvBnAct::new(
                in_channels,
                branch_features,
                1,
                1,
                0,
                1,
                true,
                ActivationKind::Relu,
                device,
            )
        });
        let branch2_in = if stride > 1 {
            in_channels
        } else {
            branch_features
        };

        Self {
            branch1_depthwise,
            branch1_pointwise,
            branch2_expand: ConvBnAct::new(
                branch2_in,
                branch_features,
                1,
                1,
                0,
                1,
                true,
                ActivationKind::Relu,
                device,
            ),
            branch2_depthwise: ConvBnAct::new(
                branch_features,
                branch_features,
                3,
                stride,
                1,
                branch_features,
                true,
                ActivationKind::None,
                device,
            ),
            branch2_linear: ConvBnAct::new(
                branch_features,
                branch_features,
                1,
                1,
                0,
                1,
                true,
                ActivationKind::Relu,
                device,
            ),
            stride,
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let branch2 = |x| {
            self.branch2_linear.forward(
                self.branch2_depthwise
                    .forward(self.branch2_expand.forward(x)),
            )
        };

        let output = if self.stride > 1 {
            let branch1 = self
                .branch1_pointwise
                .as_ref()
                .expect("stride > 1 branch")
                .forward(
                    self.branch1_depthwise
                        .as_ref()
                        .expect("stride > 1 branch")
                        .forward(input.clone()),
                );
            Tensor::cat(vec![branch1, branch2(input)], 1)
        } else {
            let chunks = input.chunk(2, 1);
            Tensor::cat(vec![chunks[0].clone(), branch2(chunks[1].clone())], 1)
        };

        channel_shuffle(output, 2)
    }
}

pub fn interpolate_nearest<B: Backend>(input: Tensor<B, 4>, size: [usize; 2]) -> Tensor<B, 4> {
    let dims = input.dims();
    if [dims[2], dims[3]] == size {
        input
    } else {
        Interpolate2dConfig::new()
            .with_output_size(Some(size))
            .with_mode(InterpolateMode::Nearest)
            .with_align_corners(false)
            .init()
            .forward(input)
    }
}

pub fn interpolate_bilinear_aligned<B: Backend>(
    input: Tensor<B, 4>,
    size: [usize; 2],
) -> Tensor<B, 4> {
    let dims = input.dims();
    if [dims[2], dims[3]] == size {
        input
    } else {
        Interpolate2dConfig::new()
            .with_output_size(Some(size))
            .with_mode(InterpolateMode::Linear)
            .with_align_corners(true)
            .init()
            .forward(input)
    }
}
