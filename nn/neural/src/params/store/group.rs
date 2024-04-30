/*
    Appellation: group <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::{Biased, Weighted};
use concision::GenerateRandom;
use ndarray::linalg::Dot;
use ndarray::{Array, Axis, Dimension, IntoDimension, Ix2, NdFloat, RemoveAxis};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num::Float;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamGroup<T = f64, D = Ix2>
where
    D: Dimension,
{
    bias: Array<T, D::Smaller>,
    features: D,
    weights: Array<T, D>,
}

impl<T, D> ParamGroup<T, D>
where
    T: Float,
    D: RemoveAxis,
{
    pub fn new(dim: impl IntoDimension<Dim = D>) -> Self {
        let dim = dim.into_dimension();
        Self {
            bias: Array::zeros(dim.remove_axis(Axis(dim.ndim() - 1))),
            features: dim.clone(),
            weights: Array::zeros(dim),
        }
    }
}

impl<T, D> ParamGroup<T, D>
where
    T: Float,
    D: Dimension,
{
    pub fn features(&self) -> &D {
        &self.features
    }

    pub fn inputs(&self) -> usize {
        self.weights.shape().last().unwrap().clone()
    }

    pub fn outputs(&self) -> usize {
        if self.features.ndim() == 1 {
            return 1;
        }
        self.weights.shape().first().unwrap().clone()
    }
}

impl<T, D> ParamGroup<T, D>
where
    D: Dimension,
    T: NdFloat,
    Self: Biased<T, Dim = D> + Weighted<T, Dim = D>,
{
    pub fn linear<D2>(&self, data: &Array<T, D2>) -> Array<T, D>
    where
        Array<T, D2>: Dot<Array<T, D>, Output = Array<T, D>>
            + std::ops::Add<Array<T, D::Smaller>, Output = Array<T, D>>,
    {
        data.dot(&self.weights().t().to_owned()) + self.bias().clone()
    }
}

impl<T, D> ParamGroup<T, D>
where
    D: Dimension + RemoveAxis,
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
        let dk = (T::one() / T::from(self.inputs()).unwrap()).sqrt();
        self.bias = Array::uniform_between(
            dk,
            self.features()
                .remove_axis(Axis(self.features().ndim() - 1))
                .clone(),
        );
        self
    }

    pub fn init_weight(mut self) -> Self {
        let dk = (T::one() / T::from(self.inputs()).unwrap()).sqrt();
        self.weights = Array::uniform_between(dk, self.features().clone());
        self
    }
}

impl<T, D> Biased<T> for ParamGroup<T, D>
where
    D: RemoveAxis,
    T: Float,
{
    type Dim = D::Smaller;

    fn bias(&self) -> &Array<T, Self::Dim> {
        &self.bias
    }

    fn bias_mut(&mut self) -> &mut Array<T, Self::Dim> {
        &mut self.bias
    }

    fn set_bias(&mut self, bias: Array<T, Self::Dim>) {
        self.bias = bias;
    }
}

impl<T, D> Weighted<T> for ParamGroup<T, D>
where
    D: Dimension,
    T: Float,
{
    type Dim = D;

    fn weights(&self) -> &Array<T, Self::Dim> {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut Array<T, Self::Dim> {
        &mut self.weights
    }

    fn set_weights(&mut self, weights: Array<T, Self::Dim>) {
        self.weights = weights;
    }
}

#[cfg(feature = "serde")]

mod impl_serde {
    use super::*;
    use serde::{Deserialize, Serialize};
    impl<'a, T, D> Deserialize<'a> for ParamGroup<T, D>
    where
        T: Deserialize<'a> + Float,
        D: Deserialize<'a> + Dimension,
        <D as Dimension>::Smaller: Deserialize<'a> + Dimension,
    {
        fn deserialize<Der>(deserializer: Der) -> Result<Self, Der::Error>
        where
            Der: serde::Deserializer<'a>,
        {
            let (bias, features, weights) = Deserialize::deserialize(deserializer)?;
            Ok(Self {
                bias,
                features,
                weights,
            })
        }
    }

    impl<T, D> Serialize for ParamGroup<T, D>
    where
        T: Float + Serialize,
        D: Dimension + RemoveAxis + Serialize,
        <D as Dimension>::Smaller: Dimension + Serialize,
    {
        fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
        where
            Ser: serde::Serializer,
        {
            (self.bias(), self.features(), self.weights()).serialize(serializer)
        }
    }
}
