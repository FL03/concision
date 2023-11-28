/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{LayerKind, LayerParams, LayerPosition, LayerShape};
use crate::func::activate::{Activate, Linear};
use crate::prelude::{Features, Forward, Neuron, Node, Parameterized, Params};
use ndarray::prelude::{Array2, Ix1, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Layer<T = f64, A = Linear>
where
    A: Activate<T>,
    T: Float,
{
    activator: A,
    pub features: LayerShape,
    name: String,
    params: LayerParams<T>,
    position: LayerPosition,
}

impl<T, A> Layer<T, A>
where
    A: Default + Activate<T>,
    T: Float,
{
    pub fn new(features: LayerShape, position: LayerPosition) -> Self {
        Self {
            activator: A::default(),
            features,
            name: String::new(),
            params: LayerParams::new(features),
            position,
        }
    }

    pub fn input(features: LayerShape) -> Self {
        Self::new(features, LayerPosition::input())
    }

    pub fn hidden(features: LayerShape, index: usize) -> Self {
        Self::new(features, LayerPosition::hidden(index))
    }

    pub fn output(features: LayerShape, index: usize) -> Self {
        Self::new(features, LayerPosition::output(index))
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T>,
    T: Float,
{
    pub fn activator(&self) -> &A {
        &self.activator
    }

    pub fn index(&self) -> usize {
        self.position().index()
    }

    pub fn kind(&self) -> &LayerKind {
        self.position().kind()
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn position(&self) -> &LayerPosition {
        &self.position
    }

    pub fn set_name(&mut self, name: impl ToString) {
        self.name = name.to_string();
    }

    pub fn set_node(&mut self, idx: usize, neuron: &Neuron<T, A>)
    where
        A: Activate<T, Ix1>,
    {
        self.params.set_node(idx, neuron.node().clone());
    }

    pub fn update_position(&mut self, idx: usize, output: bool) {
        self.position = if idx == 0 {
            LayerPosition::input()
        } else if output {
            LayerPosition::output(idx)
        } else {
            LayerPosition::hidden(idx)
        };
    }

    pub fn validate_layer(&self, other: &Self, next: bool) -> bool {
        if next {
            return self.features().inputs() == other.features().outputs();
        }
        self.features().outputs() == other.features().inputs()
    }

    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = name.to_string();
        self
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T> + Clone + 'static,
    T: Float,
{
    pub fn as_dyn(&self) -> Layer<T, Box<dyn Activate<T>>> {
        Layer {
            activator: Box::new(self.activator.clone()),
            features: self.features.clone(),
            name: self.name.clone(),
            params: self.params.clone(),
            position: self.position.clone(),
        }
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T>,
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

impl<T, A> Layer<T, A>
where
    A: Activate<T>,
    T: NdFloat,
{
    pub fn linear(&self, args: &Array2<T>) -> Array2<T> {
        args.dot(&self.params.weights().t()) + self.params.bias()
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T>,
    T: Float + SampleUniform,
{
    pub fn init(mut self, biased: bool) -> Self {
        self.params = self.params.init(biased);
        self
    }
}

impl<T, A> Forward<Array2<T>> for Layer<T, A>
where
    A: Activate<T>,
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        self.activator.activate(&self.linear(args))
    }
}

impl<T, A> Parameterized<T> for Layer<T, A>
where
    A: Activate<T>,
    T: Float,
{
    type Features = LayerShape;
    type Params = LayerParams<T>;

    fn features(&self) -> &LayerShape {
        &self.features
    }

    fn features_mut(&mut self) -> &mut LayerShape {
        &mut self.features
    }

    fn params(&self) -> &LayerParams<T> {
        &self.params
    }

    fn params_mut(&mut self) -> &mut LayerParams<T> {
        &mut self.params
    }
}

impl<T, A> PartialOrd for Layer<T, A>
where
    A: Activate<T> + PartialEq,
    T: Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.position.partial_cmp(&other.position)
    }
}

impl<T, A> From<LayerShape> for Layer<T, A>
where
    A: Activate<T> + Default,
    T: Float,
{
    fn from(features: LayerShape) -> Self {
        Self::new(features, LayerPosition::input())
    }
}

impl<T, A> IntoIterator for Layer<T, A>
where
    A: Activate<T> + Default,
    T: Float,
{
    type Item = Node<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.params.into_iter()
    }
}

impl<T, A> FromIterator<Node<T>> for Layer<T, A>
where
    A: Activate<T> + Default,
    T: Float,
{
    fn from_iter<I: IntoIterator<Item = Node<T>>>(nodes: I) -> Self {
        let params = LayerParams::from_iter(nodes);
        Self {
            activator: A::default(),
            features: *params.features(),
            name: String::new(),
            params,
            position: LayerPosition::input(),
        }
    }
}
