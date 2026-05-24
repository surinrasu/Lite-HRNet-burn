use image::RgbImage;

use super::PoseDataConfig;

#[derive(Clone, Copy, Debug)]
pub(super) struct CropWindow {
    left: f32,
    top: f32,
    width: f32,
    height: f32,
}

impl CropWindow {
    pub(super) fn from_bbox(bbox: [f32; 4], config: &PoseDataConfig) -> Self {
        let center_x = bbox[0] + bbox[2] * 0.5;
        let center_y = bbox[1] + bbox[3] * 0.5;
        let aspect = config.input_width as f32 / config.input_height as f32;

        let mut width = bbox[2];
        let mut height = bbox[3];
        if width > aspect * height {
            height = width / aspect;
        } else if width < aspect * height {
            width = height * aspect;
        }

        width *= config.bbox_padding;
        height *= config.bbox_padding;

        Self {
            left: center_x - width * 0.5,
            top: center_y - height * 0.5,
            width,
            height,
        }
    }

    pub(super) fn transform_point(&self, x: f32, y: f32, config: &PoseDataConfig) -> (f32, f32) {
        (
            (x - self.left) * config.input_width as f32 / self.width,
            (y - self.top) * config.input_height as f32 / self.height,
        )
    }

    fn source_point(&self, x: usize, y: usize, config: &PoseDataConfig) -> (f32, f32) {
        (
            self.left + (x as f32 + 0.5) * self.width / config.input_width as f32 - 0.5,
            self.top + (y as f32 + 0.5) * self.height / config.input_height as f32 - 0.5,
        )
    }
}

pub(super) fn crop_and_normalize(
    image: &RgbImage,
    crop: CropWindow,
    config: &PoseDataConfig,
) -> Vec<f32> {
    let mut output = vec![0.0; 3 * config.input_height * config.input_width];

    for y in 0..config.input_height {
        for x in 0..config.input_width {
            let (src_x, src_y) = crop.source_point(x, y, config);
            let rgb = bilinear_rgb(image, src_x, src_y);

            for (channel, pixel) in rgb.iter().enumerate() {
                let value = (*pixel / 255.0 - config.mean[channel]) / config.std[channel];
                let offset =
                    channel * config.input_height * config.input_width + y * config.input_width + x;
                output[offset] = value;
            }
        }
    }

    output
}

fn bilinear_rgb(image: &RgbImage, x: f32, y: f32) -> [f32; 3] {
    let width = image.width() as i32;
    let height = image.height() as i32;
    if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return [0.0; 3];
    }

    let x0 = x.floor() as i32;
    let y0 = y.floor() as i32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let wx = x - x0 as f32;
    let wy = y - y0 as f32;

    let p00 = image.get_pixel(x0 as u32, y0 as u32);
    let p10 = image.get_pixel(x1 as u32, y0 as u32);
    let p01 = image.get_pixel(x0 as u32, y1 as u32);
    let p11 = image.get_pixel(x1 as u32, y1 as u32);

    let mut output = [0.0; 3];
    for channel in 0..3 {
        let top = p00[channel] as f32 * (1.0 - wx) + p10[channel] as f32 * wx;
        let bottom = p01[channel] as f32 * (1.0 - wx) + p11[channel] as f32 * wx;
        output[channel] = top * (1.0 - wy) + bottom * wy;
    }
    output
}
