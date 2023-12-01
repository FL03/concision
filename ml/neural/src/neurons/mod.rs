/*
    Appellation: neurons <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # neurons
pub use self::{neuron::*, node::*, perceptron::*, synapse::*, utils::*};

pub(crate) mod neuron;
pub(crate) mod node;
pub(crate) mod perceptron;
pub(crate) mod synapse;

use crate::func::activate::Activate;
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

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::func::activate::{softmax, Activate, Softmax};
    use crate::prelude::Forward;
    // use lazy_static::lazy_static;
    use ndarray::prelude::{array, Array1, Ix1};

    fn _artificial(
        args: &Array1<f64>,
        bias: Option<Array1<f64>>,
        rho: impl Activate<f64, Ix1>,
        weights: &Array1<f64>,
    ) -> Array1<f64> {
        let bias = bias.unwrap_or_else(|| Array1::zeros(args.len()));
        let linear = args.dot(weights) + bias;
        rho.activate(&linear)
    }

    #[test]
    fn test_neuron() {
        let bias = 0.0;

        let data = array![[10.0, 10.0, 6.0, 1.0, 8.0]];
        let weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
        let neuron = Neuron::<f64, Softmax>::new(5).with_weights(weights.clone());

        let linear = data.dot(&weights) + bias;
        let exp = softmax(&linear);

        assert_eq!(exp, neuron.forward(&data));
    }

    // #[test]
    // fn test_node() {
    //     let bias = ndarray::Array1::<f64>::zeros(4);

    //     let a_data = array![10.0, 10.0, 6.0, 1.0, 8.0];
    //     let a_weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
    //     let a = Neuron::new(softmax, bias.clone(), a_weights.clone());
    //     let node_a = Node::new(a.clone()).with_data(a_data.clone());

    //     let exp = _artificial(&a_data, Some(bias.clone()), Softmax::default(), &a_weights);
    //     assert_eq!(node_a.process(), exp);

    //     let b_data = array![0.0, 9.0, 3.0, 5.0, 3.0];
    //     let b_weights = array![2.0, 8.0, 8.0, 0.0, 3.0];

    //     let b = Neuron::new(softmax, bias.clone(), b_weights.clone());
    //     let node_b = Node::new(b.clone()).with_data(b_data.clone());
    //     let exp = _artificial(&b_data, Some(bias), Softmax::default(), &b_weights);
    //     assert_eq!(node_b.process(), exp);

    //     assert_eq!(node_a.dot() + node_b.dot(), 252.0);
    // }
}
