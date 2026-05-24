use super::{PoseDataConfig, PoseSample, transform::CropWindow};

pub(super) fn generate_heatmaps(
    sample: &PoseSample,
    crop: CropWindow,
    config: &PoseDataConfig,
) -> (Vec<f32>, Vec<f32>) {
    let mut target =
        vec![0.0_f32; config.num_joints * config.heatmap_height * config.heatmap_width];
    let mut target_weight = vec![0.0_f32; config.num_joints];
    let radius = (config.sigma * 3.0).ceil() as i32;
    let denominator = 2.0 * config.sigma * config.sigma;

    for (joint_index, keypoint) in sample.keypoints.iter().take(config.num_joints).enumerate() {
        if keypoint[2] <= 0.0 {
            continue;
        }

        let (input_x, input_y) = crop.transform_point(keypoint[0], keypoint[1], config);
        let mu_x = input_x * config.heatmap_width as f32 / config.input_width as f32;
        let mu_y = input_y * config.heatmap_height as f32 / config.input_height as f32;

        if mu_x < 0.0
            || mu_y < 0.0
            || mu_x >= config.heatmap_width as f32
            || mu_y >= config.heatmap_height as f32
        {
            continue;
        }

        target_weight[joint_index] = 1.0;
        let center_x = mu_x.round() as i32;
        let center_y = mu_y.round() as i32;
        let left = (center_x - radius).max(0);
        let right = (center_x + radius).min(config.heatmap_width as i32 - 1);
        let top = (center_y - radius).max(0);
        let bottom = (center_y + radius).min(config.heatmap_height as i32 - 1);

        for y in top..=bottom {
            for x in left..=right {
                let dx = x as f32 - mu_x;
                let dy = y as f32 - mu_y;
                let value = (-(dx * dx + dy * dy) / denominator).exp();
                let offset = joint_index * config.heatmap_height * config.heatmap_width
                    + y as usize * config.heatmap_width
                    + x as usize;
                target[offset] = target[offset].max(value);
            }
        }
    }

    (target, target_weight)
}
