use std::error::Error;

use args::{Cli, Command, TrainTarget};
use cli::Parser;

mod args;
mod pose;
mod retrieval;

pub fn run() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Command::Train(args) => match args.target {
            TrainTarget::Retrieval(args) => retrieval::run_retrieval_train(args),
            TrainTarget::Pose(args) => pose::run_train(*args),
        },
        Command::Index(args) => retrieval::run_retrieval_index(args),
        Command::Search(args) => retrieval::run_retrieval_search(args),
        Command::Serve(args) => retrieval::run_retrieval_serve(args),
    }
}

#[cfg(feature = "metal")]
pub(super) fn init_metal_device() -> Result<ann::backend::wgpu::WgpuDevice, Box<dyn Error>> {
    use ann::backend::wgpu::{RuntimeOptions, WgpuDevice, graphics::Metal, init_setup};

    let device = WgpuDevice::DefaultDevice;
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let setup = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        init_setup::<Metal>(&device, RuntimeOptions::default());
    }));
    std::panic::set_hook(previous_hook);

    match setup {
        Ok(()) => Ok(device),
        Err(payload) => {
            let message = panic_payload_message(payload.as_ref());
            Err(std::io::Error::other(format!(
                "failed to initialize Burn Metal backend: {message}"
            ))
            .into())
        }
    }
}

#[cfg(feature = "metal")]
fn panic_payload_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else if let Some(message) = payload.downcast_ref::<&'static str>() {
        (*message).to_string()
    } else {
        "unknown panic".to_string()
    }
}
