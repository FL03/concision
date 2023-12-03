/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::LayerConfig;
use crate::func::activate::{Activate, Activator, Linear};
use crate::layers::{LayerParams, LayerShape};
use crate::prelude::{Features, Forward, Neuron, Node, Parameterized, Params};
use ndarray::prelude::{Array2, Ix1, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};

pub struct Layer<T = f64>
where
    T: Float,
{
    activator: Activator<T>,
    config: LayerConfig,
    params: LayerParams<T>,
}

impl<T> Layer<T>
where
    T: Float,
{
    pub fn new(activator: impl Activate<T> + 'static, config: LayerConfig) -> Self {
        let params = LayerParams::new(*config.features());
        Self {
            activator: Activator::new(Box::new(activator)),
            config,
            params,
        }
    }

    pub fn activator(&self) -> &Activator<T> {
        &self.activator
    }

    pub fn config(&self) -> &LayerConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut LayerConfig {
        &mut self.config
    }

    pub fn set_node(&mut self, idx: usize, node: &Node<T>) {
        self.params.set_node(idx, node.clone());
    }

    pub fn validate_layer(&self, other: &Self, next: bool) -> bool {
        if next {
            return self.features().inputs() == other.features().outputs();
        }
        self.features().outputs() == other.features().inputs()
    }
}

impl<T> Layer<T>
where
    T: Float + 'static,
{
    pub fn apply_gradient<F>(&mut self, gamma: T, gradient: F)
    where
        F: Fn(&Array2<T>) -> Array2<T>,
    {
        let grad = gradient(&self.params.weights());
        self.params.weights_mut().scaled_add(-gamma, &grad);
    }

    pub fn update_with_gradient(&mut self, gamma: T, grad: &Array2<T>) {
        self.params.weights_mut().scaled_add(-gamma, grad);
    }
}

impl<T> Layer<T>
where
    T: NdFloat,
{
    pub fn linear(&self, args: &Array2<T>) -> Array2<T> {
        args.dot(&self.params.weights().t()) + self.params.bias()
    }
}

impl<T> Layer<T>
where
    T: Float + SampleUniform,
{
    pub fn init(mut self, biased: bool) -> Self {
        self.params = self.params.init(biased);
        self
    }
}

impl<T> Forward<Array2<T>> for Layer<T>
where
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        self.activator.activate(&self.linear(args))
    }
}

impl<T> Parameterized<T> for Layer<T>
where
    T: Float,
{
    type Features = LayerShape;
    type Params = LayerParams<T>;

    fn features(&self) -> &LayerShape {
        self.config().features()
    }

    fn features_mut(&mut self) -> &mut LayerShape {
        self.config_mut().features_mut()
    }

    fn params(&self) -> &LayerParams<T> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut LayerParams<T> {
        &mut self.params
    }
}

// impl<T, A> PartialOrd for Layer<T, A>
// where
//     A: Activate<T> + PartialEq,
//     T: Float,
// {
//     fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
//         self.position.partial_cmp(&other.position)
//     }
// }

impl<T> From<LayerShape> for Layer<T>
where
    T: Float + 'static,
{
    fn from(features: LayerShape) -> Self {
        Self::new(Activator::linear(), features.into())
    }
}

impl<T> IntoIterator for Layer<T>
where
    T: Float,
{
    type Item = Node<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.params.into_iter()
    }
}

impl<T> FromIterator<Node<T>> for Layer<T>
where
    T: Float,
{
    fn from_iter<I: IntoIterator<Item = Node<T>>>(nodes: I) -> Self {
        let params = LayerParams::from_iter(nodes);
        Self {
            activator: Activator::linear(),
            config: LayerConfig::from(*params.features()),
            params,
        }
    }
}
