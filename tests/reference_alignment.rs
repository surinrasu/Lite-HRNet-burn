use burn::{
    backend::Flex,
    module::{Module, ModuleMapper, Param},
    tensor::{Tensor, TensorData},
};
use json::{OwnedValue, prelude::*};
use lite_hrnet_burn::{
    ConditionalChannelWeighting, CrossResolutionWeighting, HeadUpsampleMode, IterativeHead,
    LiteHrModule, LiteHrModuleType, ShuffleUnit, SpatialWeighting, Stem, channel_shuffle,
};

type B = Flex;

struct ReferenceParamMapper<B: burn::tensor::backend::Backend> {
    path: Vec<String>,
    _backend: core::marker::PhantomData<B>,
}

impl<B: burn::tensor::backend::Backend> Default for ReferenceParamMapper<B> {
    fn default() -> Self {
        Self {
            path: Vec::new(),
            _backend: core::marker::PhantomData,
        }
    }
}

impl<B: burn::tensor::backend::Backend> ReferenceParamMapper<B> {
    fn current_name(&self) -> Option<&str> {
        self.path.last().map(String::as_str)
    }

    fn value(&self) -> f64 {
        match self.current_name() {
            Some("weight") => 0.01,
            Some("gamma") | Some("running_var") => 1.0,
            Some("bias") | Some("beta") | Some("running_mean") => 0.0,
            _ => 0.0,
        }
    }
}

impl<B: burn::tensor::backend::Backend> ModuleMapper<B> for ReferenceParamMapper<B> {
    fn enter_module(&mut self, name: &str, _container_type: &str) {
        self.path.push(name.to_string());
    }

    fn exit_module(&mut self, _name: &str, _container_type: &str) {
        self.path.pop();
    }

    fn map_float<const D: usize>(&mut self, param: Param<Tensor<B, D>>) -> Param<Tensor<B, D>> {
        let (id, tensor, mapper) = param.consume();
        let dims = tensor.dims();
        let value = self.value();
        let output =
            Tensor::full(dims, value, &tensor.device()).set_require_grad(tensor.is_require_grad());
        Param::from_mapped_value(id, output, mapper)
    }
}

fn with_reference_params<M: Module<B>>(module: M) -> M {
    module.map(&mut ReferenceParamMapper::<B>::default())
}

fn input(shape: [usize; 4], scale: f32) -> Tensor<B, 4> {
    let numel = shape.iter().product::<usize>();
    let data = (0..numel)
        .map(|index| index as f32 / scale)
        .collect::<Vec<_>>();
    Tensor::from_data(TensorData::new(data, shape), &Default::default())
}

fn fixture() -> OwnedValue {
    let mut data = include_bytes!("fixtures/reference_outputs.json").to_vec();
    json::from_slice(&mut data).expect("valid fixture")
}

fn expected_tensor(case: &OwnedValue) -> (Vec<usize>, Vec<f32>) {
    let shape = case["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|value| value.as_u64().expect("usize") as usize)
        .collect::<Vec<_>>();
    let data = case["data"]
        .as_array()
        .expect("data")
        .iter()
        .map(|value| value.as_f64().expect("f32") as f32)
        .collect::<Vec<_>>();
    (shape, data)
}

fn assert_close(actual: Tensor<B, 4>, expected: &OwnedValue, tolerance: f32) {
    let actual_shape = actual.dims().to_vec();
    let actual_data = actual
        .into_data()
        .into_vec::<f32>()
        .expect("f32 tensor data");
    let (expected_shape, expected_data) = expected_tensor(expected);

    assert_eq!(actual_shape, expected_shape);
    assert_eq!(actual_data.len(), expected_data.len());
    for (index, (actual, expected)) in actual_data.iter().zip(expected_data.iter()).enumerate() {
        let diff = (actual - expected).abs();
        assert!(
            diff <= tolerance,
            "index {index}: actual {actual} expected {expected} diff {diff} > {tolerance}"
        );
    }
}

fn assert_list_close(actual: Vec<Tensor<B, 4>>, expected: &OwnedValue, tolerance: f32) {
    let expected = expected.as_array().expect("tensor list");
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected.iter()) {
        assert_close(actual, expected, tolerance);
    }
}

#[test]
fn channel_shuffle_matches_torch_reference_fixture() {
    assert_close(
        channel_shuffle(input([1, 4, 1, 2], 100.0), 2),
        &fixture()["channel_shuffle"],
        1e-6,
    );
}

#[test]
fn weighting_modules_match_torch_reference_fixture() {
    let device = Default::default();
    let fixture = fixture();

    let spatial = with_reference_params(SpatialWeighting::<B>::new(8, 4, &device));
    assert_close(
        spatial.forward(input([1, 8, 4, 3], 100.0)),
        &fixture["spatial_weighting"],
        1e-5,
    );

    let cross = with_reference_params(CrossResolutionWeighting::<B>::new(vec![4, 8], 4, &device));
    assert_list_close(
        cross.forward(vec![input([1, 4, 4, 3], 100.0), input([1, 8, 2, 2], 50.0)]),
        &fixture["cross_resolution_weighting"],
        1e-5,
    );

    let conditional = with_reference_params(ConditionalChannelWeighting::<B>::new(
        vec![8, 16],
        1,
        4,
        &device,
    ));
    assert_list_close(
        conditional.forward(vec![input([1, 8, 4, 3], 100.0), input([1, 16, 2, 2], 50.0)]),
        &fixture["conditional_channel_weighting"],
        1e-5,
    );
}

#[test]
fn stem_shuffle_module_and_head_match_torch_reference_fixture() {
    let device = Default::default();
    let fixture = fixture();

    let stem = with_reference_params(Stem::<B>::new(3, 8, 8, 1.0, &device));
    assert_close(
        stem.forward(input([1, 3, 16, 12], 100.0)),
        &fixture["stem"],
        1e-5,
    );

    let shuffle = with_reference_params(ShuffleUnit::<B>::new(8, 8, 1, &device));
    assert_close(
        shuffle.forward(input([1, 8, 4, 3], 100.0)),
        &fixture["shuffle_unit"],
        1e-5,
    );

    let module = with_reference_params(LiteHrModule::<B>::new(
        2,
        1,
        vec![8, 16],
        4,
        LiteHrModuleType::Lite,
        true,
        true,
        &device,
    ));
    assert_list_close(
        module.forward(vec![input([1, 8, 4, 4], 100.0), input([1, 16, 2, 2], 50.0)]),
        &fixture["lite_hr_module"],
        1e-5,
    );

    let head = with_reference_params(IterativeHead::<B>::new(
        vec![8, 16, 32],
        HeadUpsampleMode::BilinearAligned,
        &device,
    ));
    assert_list_close(
        head.forward(vec![
            input([1, 8, 4, 3], 100.0),
            input([1, 16, 2, 2], 50.0),
            input([1, 32, 1, 1], 25.0),
        ]),
        &fixture["iterative_head"],
        1e-5,
    );
}
