use burn::{prelude::Backend, tensor::Tensor};

pub fn joints_mse_loss<B: Backend>(
    predictions: Tensor<B, 4>,
    targets: Tensor<B, 4>,
    target_weight: Option<Tensor<B, 3>>,
) -> Tensor<B, 1> {
    let diff = predictions - targets;
    let diff = match target_weight {
        Some(weight) => {
            let [batch, joints, _] = weight.dims();
            diff * weight.reshape([batch, joints, 1, 1])
        }
        None => diff,
    };

    diff.square().mean() * 0.5
}
