/*
    Appellation: neurons <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # neurons
pub use self::{neuron::*, node::*, utils::*};

pub(crate) mod neuron;
pub(crate) mod node;

pub mod activate;

pub trait Weight {}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::activate::{Activate, Softmax, softmax};
    use super::*;
    use ndarray::{array, Array1};

    fn _artificial(
        args: &Array1<f64>,
        bias: Option<Array1<f64>>,
        rho: impl Activate<Array1<f64>>,
        weights: &Array1<f64>,
    ) -> Array1<f64> {
        rho.activate(
            args.dot(weights) - bias.unwrap_or_else(|| Array1::<f64>::zeros(args.shape()[0])),
        )
    }

    #[test]
    fn test_neuron() {
        let bias = ndarray::Array1::<f64>::zeros(4);

        let a_data = array![10.0, 10.0, 6.0, 1.0, 8.0];
        let a_weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
        let a = Neuron::new(softmax, bias.clone(), a_weights.clone());

        let exp = _artificial(&a_data, Some(bias.clone()), Softmax::default(), &a_weights);
        assert_eq!(a.compute(&a_data), exp);

        let b_data = array![0.0, 9.0, 3.0, 5.0, 3.0];
        let b_weights = array![2.0, 8.0, 8.0, 0.0, 3.0];

        let b = Neuron::new(softmax, bias.clone(), b_weights.clone());

        let exp = _artificial(&b_data, Some(bias), Softmax::default(), &b_weights);
        assert_eq!(b.compute(&b_data), exp);

        // assert_eq!(a.dot() + b.dot(), 252.0);
    }

    #[test]
    fn test_node() {
        let bias = ndarray::Array1::<f64>::zeros(4);

        let a_data = array![10.0, 10.0, 6.0, 1.0, 8.0];
        let a_weights = array![2.0, 1.0, 10.0, 1.0, 7.0];
        let a = Neuron::new(softmax, bias.clone(), a_weights.clone());
        let node_a = Node::new(a.clone()).with_data(a_data.clone());

        let exp = _artificial(&a_data, Some(bias.clone()), Softmax::default(), &a_weights);
        assert_eq!(node_a.process(), exp);

        let b_data = array![0.0, 9.0, 3.0, 5.0, 3.0];
        let b_weights = array![2.0, 8.0, 8.0, 0.0, 3.0];

        let b = Neuron::new(softmax, bias.clone(), b_weights.clone());
        let node_b = Node::new(b.clone()).with_data(b_data.clone());
        let exp = _artificial(&b_data, Some(bias), Softmax::default(), &b_weights);
        assert_eq!(node_b.process(), exp);

        assert_eq!(node_a.dot() + node_b.dot(), 252.0);
    }
}
