// directory for the training loop and related functions.

extern crate tch;
use tch::{nn, Device, Tensor};

use crate::network::StyleTransferNet;

pub struct Trainer<'a> {
    net: &'a StyleTransferNet,
    optimizer: nn::Optimizer<nn::Adam>,
}

impl<'a> Trainer<'a> {
    pub fn new(net: &'a StyleTransferNet, vs: &'a nn::VarStore) -> Trainer<'a> {
        let optimizer = nn::Adam::default().build(vs, 1e-3).unwrap();
        Trainer { net, optimizer }
    }

    pub fn train(&mut self, num_epochs: i64, device: Device) {
        for epoch in 1..=num_epochs {
            // TODO: Load your music data and style data here
            let input = Tensor::randn(&[64, 1, 28, 28], (tch::Kind::Float, device)); // Replace with actual music data
            let styles = Tensor::randn(&[64, 1, 28, 28], (tch::Kind::Float, device)); // Replace with actual style data

            let output = self.net.forward_t(&input, true);
            let loss = output.mse_loss(&styles, tch::Reduction::Mean);

            self.optimizer.backward_step(&loss);

            // Log the current loss
            println!("Epoch [{}], Loss: {:.4}", epoch, f64::from(&loss));
        }
    }
}