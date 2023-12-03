/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::LayerShape;
use crate::core::prelude::GenerateRandom;
use crate::prelude::{Biased, Features, Forward, Node, Weighted};
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2, Axis, Dimension, Ix2, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};
use std::ops;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LayerParams<T = f64> {
    bias: Array1<T>,
    pub features: LayerShape,
    weights: Array2<T>,
}

impl<T> LayerParams<T>
where
    T: Float,
{
    pub fn new(features: LayerShape) -> Self {
        Self {
            bias: Array1::zeros(features.outputs()),
            features,
            weights: Array2::zeros(features.out_by_in()),
        }
    }

    pub fn features(&self) -> &LayerShape {
        &self.features
    }

    pub fn features_mut(&mut self) -> &mut LayerShape {
        &mut self.features
    }

    pub fn set_node(&mut self, idx: usize, node: Node<T>) {
        self.bias_mut()
            .index_axis_mut(Axis(0), idx)
            .assign(&node.bias());

        self.weights_mut()
            .index_axis_mut(Axis(0), idx)
            .assign(&node.weights());
    }

    pub fn with_bias(mut self, bias: Array1<T>) -> Self {
        self.bias = bias;
        self
    }

    pub fn with_weights(mut self, weights: Array2<T>) -> Self {
        self.weights = weights;
        self
    }
}

impl<T> LayerParams<T>
where
    T: Float + 'static,
{
    pub fn update_with_gradient(&mut self, gamma: T, gradient: &Array2<T>) {
        self.weights_mut().scaled_add(-gamma, gradient);
    }
}

impl<T> LayerParams<T>
where
    T: NdFloat,
{
    pub fn reset(&mut self) {
        self.bias *= T::zero();
        self.weights *= T::zero();
    }
}

impl<T> LayerParams<T>
where
    T: Float + SampleUniform,
{
    pub fn init(mut self, biased: bool) -> Self {
        if biased {
            self = self.init_bias();
        }
        self.init_weight()
    }

    pub fn init_bias(mut self) -> Self {
        let dk = (T::one() / T::from(self.features().inputs()).unwrap()).sqrt();
        self.bias = Array1::uniform_between(dk, self.features().outputs());
        self
    }

    pub fn init_weight(mut self) -> Self {
        let dk = (T::one() / T::from(self.features().inputs()).unwrap()).sqrt();
        self.weights = Array2::uniform_between(dk, self.features().out_by_in());
        self
    }
}

impl<T> Biased<T, Ix2> for LayerParams<T>
where
    T: Float,
{
    fn bias(&self) -> &Array1<T> {
        &self.bias
    }

    fn bias_mut(&mut self) -> &mut Array1<T> {
        &mut self.bias
    }

    fn set_bias(&mut self, bias: Array1<T>) {
        self.bias = bias;
    }
}

impl<T> Weighted<T, Ix2> for LayerParams<T>
where
    T: Float,
{
    fn set_weights(&mut self, weights: Array2<T>) {
        self.weights = weights;
    }

    fn weights(&self) -> &Array2<T> {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut Array2<T> {
        &mut self.weights
    }
}

impl<T> Features for LayerParams<T>
where
    T: Float,
{
    fn inputs(&self) -> usize {
        self.features.inputs()
    }

    fn outputs(&self) -> usize {
        self.features.outputs()
    }
}

impl<T, D> Forward<Array<T, D>> for LayerParams<T>
where
    D: Dimension,
    T: NdFloat,
    Array<T, D>: Dot<Array2<T>, Output = Array<T, D>> + ops::Add<Array1<T>, Output = Array<T, D>>,
{
    type Output = Array<T, D>;

    fn forward(&self, input: &Array<T, D>) -> Self::Output {
        input.dot(&self.weights().t().to_owned()) + self.bias().clone()
    }
}

impl<T> IntoIterator for LayerParams<T>
where
    T: Float,
{
    type Item = Node<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.weights()
            .axis_iter(Axis(0))
            .zip(self.bias().axis_iter(Axis(0)))
            .map(|(w, b)| (w.to_owned(), b.to_owned()).into())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<T> FromIterator<Node<T>> for LayerParams<T>
where
    T: Float,
{
    fn from_iter<I: IntoIterator<Item = Node<T>>>(nodes: I) -> Self {
        let nodes = nodes.into_iter().collect::<Vec<_>>();
        let mut iter = nodes.iter();
        let node = iter.next().unwrap();
        let shape = LayerShape::new(*node.features(), nodes.len());
        let mut params = LayerParams::new(shape);
        params.set_node(0, node.clone());
        for (i, node) in iter.into_iter().enumerate() {
            params.set_node(i + 1, node.clone());
        }
        params
    }
}
