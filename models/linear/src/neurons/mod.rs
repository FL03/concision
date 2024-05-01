/*
    Appellation: neurons <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # neurons
pub use self::{node::*, perceptron::*};

pub(crate) mod node;
pub(crate) mod perceptron;

use crate::neural::func::activate::Activate;
use ndarray::prelude::{Array0, Array1, Array2, Ix1, NdFloat};

pub trait ArtificialNeuron<T>
where
    T: NdFloat,
{
    type Rho: Activate<T, Ix1>;

    fn bias(&self) -> Array0<T>;

    fn linear(&self, args: &Array2<T>) -> Array1<T> {
        args.dot(self.weights()) + self.bias()
    }

    fn forward(&self, args: &Array2<T>) -> Array1<T> {
        self.rho().activate(&self.linear(args))
    }

    fn rho(&self) -> &Self::Rho;

    fn weights(&self) -> &Array1<T>;
}

#[cfg(test)]
mod tests {
    use super::Perceptron;
    use concision::traits::Forward;
    use ndarray::*;
    use neural::prelude::{softmax, Softmax};

    #[test]
    fn test_neuron() {
        let bias = 0.0;

        let data = array![[10.0, 10.0, 6.0, 1.0, 8.0]];
        let weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
        let neuron = Perceptron::<f64, Softmax>::new(5).with_weights(weights.clone());

        let linear = data.dot(&weights) + bias;
        let exp = softmax(&linear);

        assert_eq!(exp, neuron.forward(&data));
    }
}
