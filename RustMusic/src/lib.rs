// main.rs directory to initialize the network, create a trainer, and start the training process.

mod network;
mod train;

use tch::{Device, nn};

use crate::network::StyleTransferNet;
use crate::train::Trainer;

fn main() {
    // Set the device (CPU or CUDA)
    let device = Device::cuda_if_available();
    let vs = nn::VarStore::new(device);
    
    // Create the neural style transfer network
    let net = StyleTransferNet::new(&vs.root());

    // Define your hyperparameters
    let num_epochs = 10; // Number of epochs for training

    // Create a Trainer instance
    let mut trainer = Trainer::new(&net, &vs);

    // Train the network (you'd replace 10 with the actual number of epochs you desire)
    trainer.train(10, device);

    vs.save("style_transfer_model.pt").expect("Could not save the model");
}