/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::LayerShape;
use crate::core::prelude::GenerateRandom;
use crate::prelude::{Features, Forward, Node};
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2, Axis, Dimension, NdFloat};
use ndarray::{LinalgScalar, ScalarOperand};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num::traits::{Float, NumAssignOps};
use serde::{Deserialize, Serialize};
use std::ops::{self, Neg};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LayerParams<T = f64> {
    bias: Option<Array1<T>>,
    pub features: LayerShape,
    weights: Array2<T>,
}

impl<T> LayerParams<T> {
    pub fn bias(&self) -> &Option<Array1<T>> {
        &self.bias
    }

    pub fn bias_mut(&mut self) -> &mut Option<Array1<T>> {
        &mut self.bias
    }

    pub fn features(&self) -> &LayerShape {
        &self.features
    }

    pub fn features_mut(&mut self) -> &mut LayerShape {
        &mut self.features
    }

    pub fn is_biased(&self) -> bool {
        self.bias.is_some()
    }

    pub fn set_bias(&mut self, bias: Option<Array1<T>>) {
        self.bias = bias;
    }

    pub fn set_weights(&mut self, weights: Array2<T>) {
        self.weights = weights;
    }

    pub fn weights(&self) -> &Array2<T> {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut Array2<T> {
        &mut self.weights
    }

    pub fn with_bias(mut self, bias: Option<Array1<T>>) -> Self {
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
    T: Default,
{
    pub fn new(features: LayerShape) -> Self {
        Self::create(false, features)
    }

    pub fn create(biased: bool, features: LayerShape) -> Self {
        let bias = if biased {
            Some(Array1::default(features.outputs()))
        } else {
            None
        };
        Self {
            bias,
            features,
            weights: Array2::default(features.out_by_in()),
        }
    }

    pub fn biased(features: LayerShape) -> Self {
        Self::create(true, features)
    }

    pub fn reset(&mut self)
    where
        T: NumAssignOps + ScalarOperand,
    {
        if let Some(bias) = self.bias() {
            self.bias = Some(Array1::default(bias.dim()));
        }
        self.weights *= T::default();
    }

    pub fn set_node(&mut self, idx: usize, node: Node<T>)
    where
        T: Clone,
    {
        if let Some(bias) = node.bias() {
            if !self.is_biased() {
                let mut tmp = Array1::default(self.features().outputs());
                tmp.index_axis_mut(Axis(0), idx).assign(bias);
                self.bias = Some(tmp);
            }
            self.bias
                .as_mut()
                .unwrap()
                .index_axis_mut(Axis(0), idx)
                .assign(bias);
        }

        self.weights_mut()
            .index_axis_mut(Axis(0), idx)
            .assign(&node.weights());
    }
}

impl<T> LayerParams<T>
where
    T: LinalgScalar + Neg<Output = T> + 'static,
{
    pub fn update_with_gradient(&mut self, gamma: T, gradient: &Array2<T>) {
        self.weights_mut().scaled_add(-gamma, gradient);
    }
}

impl<T> LayerParams<T>
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
        let dk = (T::one() / T::from(self.features().inputs()).unwrap()).sqrt();
        self.bias = Some(Array1::uniform_between(dk, self.features().outputs()));
        self
    }

    pub fn init_weight(mut self) -> Self {
        let dk = (T::one() / T::from(self.features().inputs()).unwrap()).sqrt();
        self.weights = Array2::uniform_between(dk, self.features().out_by_in());
        self
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
        let w = self.weights().t().to_owned();
        if let Some(bias) = self.bias() {
            return input.dot(&w) + bias.clone();
        }
        input.dot(&w)
    }
}

impl<T> IntoIterator for LayerParams<T>
where
    T: Clone,
{
    type Item = Node<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        if let Some(bias) = self.bias() {
            return self
                .weights()
                .axis_iter(Axis(0))
                .zip(bias.axis_iter(Axis(0)))
                .map(|(w, b)| (w.to_owned(), b.to_owned()).into())
                .collect::<Vec<_>>()
                .into_iter();
        }
        self.weights()
            .axis_iter(Axis(0))
            .map(|w| (w.to_owned(), None).into())
            .collect::<Vec<_>>()
            .into_iter()
    }
}

impl<T> FromIterator<Node<T>> for LayerParams<T>
where
    T: Clone + Default,
{
    fn from_iter<I: IntoIterator<Item = Node<T>>>(nodes: I) -> Self {
        let nodes = nodes.into_iter().collect::<Vec<_>>();
        let mut iter = nodes.iter();
        let node = iter.next().unwrap();
        let shape = LayerShape::new(node.features(), nodes.len());
        let mut params = LayerParams::create(true, shape);
        params.set_node(0, node.clone());
        for (i, node) in iter.into_iter().enumerate() {
            params.set_node(i + 1, node.clone());
        }
        params
    }
}
