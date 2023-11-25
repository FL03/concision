/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{LayerKind, LayerParams, LayerPosition, LayerShape};
use crate::prelude::{Activate, Forward, Linear, Neuron, Parameterized, Params};
use ndarray::prelude::{Array2, Axis, Ix1, Ix2, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Layer<T = f64, A = Linear>
where
    A: Activate<T, Ix2>,
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
    A: Default + Activate<T, Ix2>,
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
    A: Activate<T, Ix2>,
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

    pub fn update_position(&mut self, idx: usize, output: bool) {
        self.position = if idx == 0 {
            LayerPosition::input()
        } else if output {
            LayerPosition::output(idx)
        } else {
            LayerPosition::hidden(idx)
        };
    }

    pub fn validate_layer(&self, other: &Self) -> bool {
        let pos = self.position().index().abs_diff(other.position().index());
        if pos == 1 {
            if self.position().index() > other.position().index() {
                return self.features().inputs() == other.features().outputs();
            } else {
                return self.features().outputs() == other.features().inputs();
            }
        }
        false
    }

    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = name.to_string();
        self
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T, Ix2> + Clone + 'static,
    T: Float,
{
    pub fn as_dyn(&self) -> Layer<T, Box<dyn Activate<T, Ix2>>> {
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
    A: Activate<T, Ix2>,
    T: Float + 'static,
{
    pub fn update_with_gradient(&mut self, gamma: T, grad: &Array2<T>) {
        self.params.weights_mut().scaled_add(-gamma, grad);
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T, Ix2>,
    T: NdFloat,
{
    pub fn linear(&self, args: &Array2<T>) -> Array2<T> {
        args.dot(&self.params.weights().t()) + self.params.bias()
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<T, Ix2>,
    T: Float + SampleUniform,
{
    pub fn init(mut self, biased: bool) -> Self {
        self.params = self.params.init(biased);
        self
    }
}

impl<T, A> Forward<Array2<T>> for Layer<T, A>
where
    A: Activate<T, Ix2>,
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        self.activator.activate(&self.linear(args))
    }
}

impl<T, A> Parameterized<T> for Layer<T, A>
where
    A: Activate<T, Ix2>,
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
    A: Activate<T, Ix2> + PartialEq,
    T: Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.position.partial_cmp(&other.position)
    }
}

impl<T, A> From<LayerShape> for Layer<T, A>
where
    A: Activate<T, Ix2> + Default,
    T: Float,
{
    fn from(features: LayerShape) -> Self {
        Self::new(features, LayerPosition::input())
    }
}

// impl<T, A> IntoIterator for Layer<T, A>
// where
//     A: Activate<T, Ix2>,
//     T: Float,
// {
//     type Item = (Array1<T>, Array0<T>);
//     type IntoIter = std::vec::IntoIter<Self::Item>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.params().weights().axis_iter(Axis(0)).zip(self.params().bias().axis_iter(Axis(0))).map(|(w, b)| (w.to_owned(), b.to_owned())).collect::<Vec<_>>().into_iter()
//     }
// }

impl<T, A> IntoIterator for Layer<T, A>
where
    A: Activate<T, Ix2> + Activate<T, Ix1> + Default,
    T: Float,
{
    type Item = Neuron<T, A>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.params()
            .weights()
            .axis_iter(Axis(0))
            .zip(self.params().bias().axis_iter(Axis(0)))
            .map(|(w, b)| (w.to_owned(), b.to_owned()).into())
            .collect::<Vec<_>>()
            .into_iter()
    }
}
