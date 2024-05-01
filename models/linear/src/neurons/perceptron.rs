/*
    Appellation: perceptron <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Node;
use concision::prelude::Forward;

use ndarray::prelude::{Array0, Array1, Array2, NdFloat};
use ndrand::rand_distr::uniform::SampleUniform;
use ndrand::rand_distr::{Distribution, StandardNormal};
use num::Float;

pub fn linear_activation<T: Clone>(x: &Array1<T>) -> Array1<T> {
    x.clone()
}

pub type BasicPerceptron<T> = Perceptron<T, dyn Fn(&Array1<T>) -> Array1<T>>;

/// Artificial Neuron
#[derive(Clone, Debug, PartialEq)]
pub struct Perceptron<T, F = Box<dyn Fn(&Array1<T>) -> Array1<T>>> {
    activation: F,
    node: Node<T>,
}

impl<T, F> Perceptron<T, F>
where
    T: Float,
{
    pub fn new(activation: F, features: usize) -> Self {
        Self {
            activation,
            node: Node::create(false, features),
        }
    }

    pub fn activate(&self, x: &Array1<T>) -> Array1<T>
    where
        F: for<'a> Fn(&'a Array1<T>) -> Array1<T>,
    {
        (self.activation)(x)
    }

    pub fn node(&self) -> &Node<T> {
        &self.node
    }

    pub fn node_mut(&mut self) -> &mut Node<T> {
        &mut self.node
    }

    pub fn features(&self) -> usize {
        self.node().features()
    }

    pub fn params(&self) -> &Node<T> {
        &self.node
    }

    pub fn params_mut(&mut self) -> &mut Node<T> {
        &mut self.node
    }

    pub fn rho(&self) -> &F {
        &self.activation
    }

    pub fn set_weights(&mut self, weights: Array1<T>) {
        self.node.set_weights(weights);
    }

    pub fn weights(&self) -> &Array1<T> {
        self.node.weights()
    }

    pub fn weights_mut(&mut self) -> &mut Array1<T> {
        self.node.weights_mut()
    }

    pub fn with_bias(self, bias: Option<Array0<T>>) -> Self {
        Self {
            node: self.node.with_bias(bias),
            ..self
        }
    }

    pub fn with_node(self, node: Node<T>) -> Self {
        Self { node, ..self }
    }

    pub fn with_rho<G>(self, rho: G) -> Perceptron<T, G> {
        Perceptron {
            activation: rho,
            node: self.node,
        }
    }

    pub fn with_weights(self, weights: Array1<T>) -> Self {
        Self {
            node: self.node.with_weights(weights),
            ..self
        }
    }

    pub fn apply_gradient<G>(&mut self, gamma: T, gradient: G)
    where
        G: Fn(&Array1<T>) -> Array1<T>,
        T: 'static,
    {
        let grad = gradient(self.node().weights());
        self.update_with_gradient(gamma, &grad);
    }

    pub fn update_with_gradient(&mut self, gamma: T, grad: &Array1<T>)
    where
        T: 'static,
    {
        self.node.weights_mut().scaled_add(-gamma, grad);
    }
}

impl<T, A> Perceptron<T, A>
where
    T: Float + SampleUniform,
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

impl<T, A> Forward<Array2<T>> for Perceptron<T, A>
where
    T: NdFloat,
    A: Fn(&Array1<T>) -> Array1<T>,
{
    type Output = Array1<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        let linstep = self.params().forward(args);
        self.activate(&linstep)
    }
}

macro_rules! impl_into_perceptron {
    ($(($w:ty, $b:ty)),* $(,)?) => {
        $(
            impl_into_perceptron!(@impl $w, $b);
        )*
    };
    (@impl $w:ty, $b:ty) => {
        impl<T> From<($w, $b)> for Perceptron<T>
        where
            T: Clone + nd::NdFloat + 'static,
        {
            fn from((weights, bias): ($w, $b)) -> Self {
                Self {
                    activation: Box::new(linear_activation),
                    node: Node::from((weights, bias)),
                }
            }
        }

    };
    (@into $w:ty, $b:ty) => {

        impl<T, A> From<Perceptron<T, A>> for ($w, $b)
        where
            T: Clone + nd::NdFloat + 'static,
        {
            fn from(neuron: Perceptron<T, A>) -> Self {
                neuron.node().clone().into()
            }
        }

    };

}

impl_into_perceptron!((Array1<T>, Array0<T>), (Array1<T>, T));

impl<T, A> From<(Array1<T>, Array0<T>, A)> for Perceptron<T, A>
where
    T: Float,
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
{
    fn from((weights, bias, activation): (Array1<T>, T, A)) -> Self {
        Self {
            activation,
            node: Node::from((weights, bias)),
        }
    }
}
