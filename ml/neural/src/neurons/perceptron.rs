/*
    Appellation: neuron <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Node;
use crate::func::activate::{Activate, Linear};
use crate::prelude::{Forward, Parameterized, ParameterizedExt, Weighted};
use ndarray::prelude::{Array0, Array1, Array2, Ix1, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num::Float;

/// Artificial Neuron
#[derive(Clone, Debug, PartialEq)]
pub struct Perceptron<T = f64, A = Linear>
where
    A: Activate<T, Ix1>,
    T: Float,
{
    activation: A,
    node: Node<T>,
}

impl<T, A> Perceptron<T, A>
where
    A: Activate<T, Ix1>,
    T: Float,
{
    pub fn node(&self) -> &Node<T> {
        &self.node
    }

    pub fn node_mut(&mut self) -> &mut Node<T> {
        &mut self.node
    }

    pub fn rho(&self) -> &A {
        &self.activation
    }

    pub fn with_bias(mut self, bias: Array0<T>) -> Self {
        self.node = self.node.with_bias(bias);
        self
    }

    pub fn with_rho<B: Activate<T, Ix1>>(self, rho: B) -> Perceptron<T, B> {
        Perceptron {
            activation: rho,
            node: self.node,
        }
    }

    pub fn with_node(mut self, node: Node<T>) -> Self {
        self.node = node;
        self
    }

    pub fn with_weights(mut self, weights: Array1<T>) -> Self {
        self.node = self.node.with_weights(weights);
        self
    }
}

impl<T, A> Perceptron<T, A>
where
    T: NdFloat,
    A: Activate<T, Ix1> + Default,
{
    pub fn new(features: usize) -> Self {
        Self {
            activation: A::default(),
            node: Node::new(features),
        }
    }
}

impl<T, A> Perceptron<T, A>
where
    T: NdFloat,
    A: Activate<T, Ix1>,
{
    pub fn apply_gradient<G>(&mut self, gamma: T, gradient: G)
    where
        G: Fn(&Array1<T>) -> Array1<T>,
    {
        let grad = gradient(self.node().weights());
        self.update_with_gradient(gamma, &grad);
    }

    pub fn update_with_gradient(&mut self, gamma: T, grad: &Array1<T>) {
        self.node.weights_mut().scaled_add(-gamma, grad);
    }
}

impl<T, A> Perceptron<T, A>
where
    T: Float + SampleUniform,
    A: Activate<T, Ix1>,
    StandardNormal: Distribution<T>,
{
    pub fn init(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        self.node = self.node.init_bias();
        self
    }

    pub fn init_weight(mut self) -> Self {
        self.node = self.node.init_weight();
        self
    }
}

// impl<T, A> Biased<T, Ix1> for Neuron<T, A>
// where
//     T: Float,
//     A: Activate<T, Ix1>,
// {
//     fn bias(&self) -> &Array0<T> {
//         self.node.bias()
//     }

//     fn bias_mut(&mut self) -> &mut Array0<T> {
//         self.node.bias_mut()
//     }

//     fn set_bias(&mut self, bias: Array0<T>) {
//         self.node.set_bias(bias);
//     }
// }

impl<T, A> Weighted<T, Ix1> for Perceptron<T, A>
where
    T: Float,
    A: Activate<T, Ix1>,
{
    fn weights(&self) -> &Array1<T> {
        self.node.weights()
    }

    fn weights_mut(&mut self) -> &mut Array1<T> {
        self.node.weights_mut()
    }

    fn set_weights(&mut self, weights: Array1<T>) {
        self.node.set_weights(weights);
    }
}

impl<T, A> Parameterized<T, Ix1> for Perceptron<T, A>
where
    A: Activate<T, Ix1>,
    T: Float,
{
    type Features = usize;

    type Params = Node<T>;

    fn features(&self) -> &Self::Features {
        self.node.features()
    }

    fn features_mut(&mut self) -> &mut Self::Features {
        self.node.features_mut()
    }

    fn params(&self) -> &Self::Params {
        &self.node
    }

    fn params_mut(&mut self) -> &mut Self::Params {
        &mut self.node
    }
}

// impl Forward<Array1<f64>> for Neuron {
//     type Output = f64;

//     fn forward(&self, args: &Array1<f64>) -> Self::Output {
//         self.rho().activate(args.dot(&self.weights().t().to_owned()) + self.bias)
//     }

// }

impl<T, A> Forward<Array2<T>> for Perceptron<T, A>
where
    T: NdFloat,
    A: Activate<T, Ix1>,
{
    type Output = Array1<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        let linstep = args.dot(&self.node().weights().t()) + self.bias();
        self.rho().activate(&linstep)
    }
}

impl<T, A> From<(Array1<T>, Array0<T>)> for Perceptron<T, A>
where
    T: Float,
    A: Activate<T, Ix1> + Default,
{
    fn from((weights, bias): (Array1<T>, Array0<T>)) -> Self {
        Self {
            activation: A::default(),
            node: Node::from((weights, bias)),
        }
    }
}

impl<T, A> From<(Array1<T>, T)> for Perceptron<T, A>
where
    T: NdFloat,
    A: Activate<T, Ix1> + Default,
{
    fn from((weights, bias): (Array1<T>, T)) -> Self {
        Self {
            activation: A::default(),
            node: Node::from((weights, bias)),
        }
    }
}

impl<T, A> From<(Array1<T>, Array0<T>, A)> for Perceptron<T, A>
where
    T: Float,
    A: Activate<T, Ix1>,
{
    fn from((weights, bias, activation): (Array1<T>, Array0<T>, A)) -> Self {
        Self {
            activation,
            node: Node::from((weights, bias)),
        }
    }
}

impl<T, A> From<(Array1<T>, T, A)> for Perceptron<T, A>
where
    T: NdFloat,
    A: Activate<T, Ix1>,
{
    fn from((weights, bias, activation): (Array1<T>, T, A)) -> Self {
        Self {
            activation,
            node: Node::from((weights, bias)),
        }
    }
}

impl<T, A> From<Perceptron<T, A>> for (Array1<T>, Array0<T>)
where
    T: Float,
    A: Activate<T, Ix1>,
{
    fn from(neuron: Perceptron<T, A>) -> Self {
        neuron.node().clone().into()
    }
}
