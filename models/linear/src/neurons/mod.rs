/*
    Appellation: neurons <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # neurons
pub use self::{node::*, perceptron::*};

pub(crate) mod node;
pub(crate) mod perceptron;

pub trait ArtificialNeuron {
    type Rho: for<'a> Fn(&'a Self::Output) -> Self::Output;
    type Output;

    fn activate(&self, x: &Self::Output) -> Self::Output {
        (self.rho())(x)
    }

    fn rho(&self) -> &Self::Rho;
}

#[cfg(test)]
mod tests {
    use super::Perceptron;
    use concision::func::activate::softmax;
    use concision::traits::Forward;
    use ndarray::*;

    #[test]
    fn test_neuron() {
        let bias = 0.0;

        let data = array![[10.0, 10.0, 6.0, 1.0, 8.0]];
        let weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
        let neuron = Perceptron::<f64>::new(Box::new(softmax), 5).with_weights(weights.clone());

        let linear = data.dot(&weights) + bias;
        let exp = softmax(&linear);

        assert_eq!(exp, neuron.forward(&data));
    }
}
