use super::InputSize;

pub(super) fn parse_input_size(value: &str) -> Result<InputSize, String> {
    let (height, width) = value
        .split_once('x')
        .or_else(|| value.split_once('X'))
        .ok_or_else(|| "expected HEIGHTxWIDTH, for example 256x192".to_string())?;
    let height = parse_positive_usize(height, "height")?;
    let width = parse_positive_usize(width, "width")?;

    Ok(InputSize { height, width })
}

fn parse_positive_usize(value: &str, label: &str) -> Result<usize, String> {
    let parsed = value
        .parse::<usize>()
        .map_err(|error| format!("invalid {label} `{value}`: {error}"))?;
    if parsed == 0 {
        Err(format!("{label} must be greater than 0"))
    } else {
        Ok(parsed)
    }
}

pub(super) fn parse_positive_count(value: &str) -> Result<usize, String> {
    parse_positive_usize(value, "value")
}

pub(super) fn parse_positive_f64(value: &str) -> Result<f64, String> {
    let parsed = value
        .parse::<f64>()
        .map_err(|error| format!("invalid float `{value}`: {error}"))?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err("value must be a finite number greater than 0".to_string())
    }
}

pub(super) fn parse_positive_f32(value: &str) -> Result<f32, String> {
    let parsed = value
        .parse::<f32>()
        .map_err(|error| format!("invalid float `{value}`: {error}"))?;
    if parsed.is_finite() && parsed > 0.0 {
        Ok(parsed)
    } else {
        Err("value must be a finite number greater than 0".to_string())
    }
}
