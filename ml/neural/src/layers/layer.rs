/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Features, LayerParams, LayerType, Position};
use crate::prelude::{Activate, Forward, LinearActivation, Parameterized, Params};
use ndarray::prelude::{Array2, NdFloat};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Layer<T = f64, A = LinearActivation>
where
    A: Activate<Array2<T>>,
    T: Float,
{
    activator: A,
    pub features: Features,
    params: LayerParams<T>,
    position: Position,
}

impl<T, A> Layer<T, A>
where
    A: Activate<Array2<T>> + Default,
    T: Float,
{
    pub fn new(features: Features, position: Position) -> Self {
        Self {
            activator: A::default(),
            features,
            params: LayerParams::new(features),
            position,
        }
    }

    pub fn new_input(features: Features) -> Self {
        Self::new(features, Position::input())
    }

    pub fn hidden(features: Features, index: usize) -> Self {
        Self::new(features, Position::hidden(index))
    }

    pub fn output(features: Features, index: usize) -> Self {
        Self::new(features, Position::output(index))
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<Array2<T>>,
    T: Float,
{
    pub fn activator(&self) -> &A {
        &self.activator
    }

    pub fn kind(&self) -> &LayerType {
        self.position().kind()
    }

    pub fn position(&self) -> &Position {
        &self.position
    }

    pub fn set_position(&mut self, position: Position) {
        self.position = position;
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<Array2<T>>,
    T: Float + 'static,
{
    pub fn update_with_gradient(&mut self, gamma: T, grad: &Array2<T>) {
        self.params.weights_mut().scaled_add(-gamma, grad);
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<Array2<T>>,
    T: NdFloat,
{
    pub fn linear(&self, args: &Array2<T>) -> Array2<T> {
        args.dot(&self.params.weights().t()) + self.params.bias()
    }
}

impl<T, A> Layer<T, A>
where
    A: Activate<Array2<T>>,
    T: Float + SampleUniform,
{
    pub fn init(mut self, biased: bool) -> Self {
        self.params = self.params.init(biased);
        self
    }
}

impl<T, A> Forward<Array2<T>> for Layer<T, A>
where
    A: Activate<Array2<T>>,
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        self.activator.activate(self.linear(args))
    }
}

impl<T, A> Parameterized<T> for Layer<T, A>
where
    A: Activate<Array2<T>>,
    T: Float,
{
    type Features = Features;
    type Params = LayerParams<T>;

    fn features(&self) -> &Features {
        &self.features
    }

    fn features_mut(&mut self) -> &mut Features {
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
    A: Activate<Array2<T>> + PartialEq,
    T: Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.position.partial_cmp(&other.position)
    }
}

impl<T, A> From<Features> for Layer<T, A>
where
    A: Activate<Array2<T>> + Default,
    T: Float,
{
    fn from(features: Features) -> Self {
        Self::new(features, Position::input())
    }
}
