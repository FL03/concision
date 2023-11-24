/*
    Appellation: group <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Biased, Weighted};
use ndarray::prelude::{Array, Axis, Dimension, Ix2};
use ndarray::{IntoDimension, RemoveAxis};
use num::Float;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamGroup<T = f64, D = Ix2>
where
    T: Float,
    D: Dimension,
    <D as Dimension>::Smaller: Dimension,
{
    bias: Array<T, D::Smaller>,
    weights: Array<T, D>,
}

impl<T, D> ParamGroup<T, D>
where
    T: Float,
    D: Dimension + RemoveAxis,
    <D as Dimension>::Smaller: Dimension,
{
    pub fn new(dim: impl IntoDimension<Dim = D>) -> Self {
        let dim = dim.into_dimension();
        let smaller = dim.clone().remove_axis(Axis(dim.ndim() - 1));
        Self {
            bias: Array::zeros(smaller),
            weights: Array::zeros(dim),
        }
    }
}

impl<T, D> Biased<T, D> for ParamGroup<T, D>
where
    T: Float,
    D: Dimension + RemoveAxis,
    <D as Dimension>::Smaller: Dimension,
{
    fn bias(&self) -> &Array<T, D::Smaller> {
        &self.bias
    }

    fn bias_mut(&mut self) -> &mut Array<T, D::Smaller> {
        &mut self.bias
    }

    fn set_bias(&mut self, bias: Array<T, D::Smaller>) {
        self.bias = bias;
    }
}

impl<T, D> Weighted<T, D> for ParamGroup<T, D>
where
    T: Float,
    D: Dimension + RemoveAxis,
    <D as Dimension>::Smaller: Dimension,
{
    fn weights(&self) -> &Array<T, D> {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut Array<T, D> {
        &mut self.weights
    }

    fn set_weights(&mut self, weights: Array<T, D>) {
        self.weights = weights;
    }
}
