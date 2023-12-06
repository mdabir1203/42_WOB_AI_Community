//directory for the neural network architecture.

extern crate tch;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Tensor};


#[derive(Debug)]
pub struct StyleTransferNet {
    // Example layers
    conv1: nn::Conv2d,
    conv2: nn::Conv2d,
    fc1: nn::Linear,
    fc2: nn::Linear,

//
//It defines the necessary CNN and dense layers
//Layers are stacked appropriately, with outputs from one feeding into the next
//These layers are then assigned as fields in the StyleTransferNet module


impl StyleTransferNet {
    pub fn new(vs: &nn::Path) -> StyleTransferNet {
        // Define the layers and their parameters
        let conv1 = nn::conv2d(vs, 1, 32, 5, Default::default());
        let conv2 = nn::conv2d(vs, 32, 64, 5, Default::default());
        let fc1 = nn::linear(vs, 64 * 4 * 4, 128, Default::default());
        let fc2 = nn::linear(vs, 128, 10, Default::default()); // Adapt the output size to your needs

        StyleTransferNet {
            conv1,
            conv2,
            fc1,
            fc2,
        }
    }
}

// multiple CNN layers to hierarchically learn spatial features, 
// max pooling for some translation invariance,
// flattening and adding fully connected layers at the end for the classification.

impl ModuleT for StyleTransferNet {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Tensor {
        xs.view([-1, 1, 28, 28]) // Example input reshape; adapt to your music data
            .apply(&self.conv1)
            .max_pool2d_default(2)
            .relu()
            .apply(&self.conv2)
            .max_pool2d_default(2)
            .relu()
            .view([-1, 64 * 4 * 4]) // Flatten the output
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
    }
}