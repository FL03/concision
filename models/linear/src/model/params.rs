/*
    Appellation: group <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::kinds::*;

pub(crate) mod kinds;

use crate::model::Features;
use crate::{Node, Weighted};
use concision::error::PredictError;
use concision::{GenerateRandom, Predict};
use core::ops;
use nd::linalg::Dot;
use nd::*;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num::{Float, One, Zero};

#[cfg(no_std)]
use alloc::vec;
#[cfg(feature = "std")]
use std::vec;

fn build_bias<T, F, D>(biased: bool, dim: D, builder: F) -> Option<Array<T, D::Smaller>>
where
    D: RemoveAxis,
    F: Fn(D::Smaller) -> Array<T, D::Smaller>,
{
    if biased {
        Some(builder(dim.remove_axis(Axis(dim.ndim() - 1))))
    } else {
        None
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LinearParams<T = f64, D = Ix2>
where
    D: Dimension,
{
    bias: Option<Array<T, D::Smaller>>,
    features: D,
    weights: Array<T, D>,
}

impl<T, D> LinearParams<T, D>
where
    D: RemoveAxis,
{
    pub fn new(biased: bool, dim: impl IntoDimension<Dim = D>) -> Self
    where
        T: Clone + Default,
    {
        let dim = dim.into_dimension();
        let bias = build_bias(biased, dim.clone(), |dim| Array::default(dim));
        Self {
            bias,
            features: dim.clone(),
            weights: Array::default(dim),
        }
    }

    pub fn ones<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        T: Clone + One,
    {
        let shape = shape.into_shape();
        let dim = shape.raw_dim().clone();
        let bias = build_bias(true, dim.clone(), |dim| Array::ones(dim));
        Self {
            bias,
            features: dim.clone(),
            weights: Array::ones(dim),
        }
    }

    pub fn zeros<Sh>(shape: Sh) -> Self
    where
        Sh: ShapeBuilder<Dim = D>,
        T: Clone + Zero,
    {
        let shape = shape.into_shape();
        let dim = shape.raw_dim().clone();
        let bias = build_bias(true, dim.clone(), |dim| Array::zeros(dim));
        Self {
            bias,
            features: dim.clone(),
            weights: Array::zeros(dim),
        }
    }

    pub fn bias(&self) -> Option<&Array<T, D::Smaller>> {
        self.bias.as_ref()
    }

    pub fn bias_mut(&mut self) -> Option<&mut Array<T, D::Smaller>> {
        self.bias.as_mut()
    }

    pub fn features(&self) -> &D {
        &self.features
    }

    pub fn inputs(&self) -> usize {
        self.weights.shape().last().unwrap().clone()
    }

    pub fn is_biased(&self) -> bool {
        self.bias.is_some()
    }

    pub fn linear<A, B>(&self, data: &A) -> B
    where
        A: Dot<Array<T, D>, Output = B>,
        B: for<'a> ops::Add<&'a Array<T, D::Smaller>, Output = B>,
        T: NdFloat,
    {
        let dot = data.dot(&self.weights().t().to_owned());
        if let Some(bias) = self.bias() {
            return dot + bias;
        }
        dot
    }

    pub fn outputs(&self) -> usize {
        if self.features.ndim() == 1 {
            return 1;
        }
        self.weights.shape().first().unwrap().clone()
    }

    pub fn weights(&self) -> &Array<T, D> {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut Array<T, D> {
        &mut self.weights
    }
}

impl<T> LinearParams<T>
where
    T: Float,
{
    pub fn set_node(&mut self, idx: usize, node: Node<T>) {
        if let Some(bias) = node.bias() {
            if !self.is_biased() {
                let mut tmp = Array1::zeros(self.outputs());
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

impl<T, D> LinearParams<T, D>
where
    D: RemoveAxis,
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
        let dim = self
            .features()
            .remove_axis(Axis(self.features().ndim() - 1));
        self.bias = Some(Array::uniform_between(dk, dim));
        self
    }

    pub fn init_weight(mut self) -> Self {
        let dk = (T::one() / T::from(self.inputs()).unwrap()).sqrt();
        self.weights = Array::uniform_between(dk, self.features().clone());
        self
    }
}

// impl<T, D> Biased<T> for ParamGroup<T, D>
// where
//     D: RemoveAxis,
//     T: Float,
// {
//     type Dim = D::Smaller;

//     fn bias(&self) -> &Array<T, Self::Dim> {
//         self.bias.as_ref().unwrap()
//     }

//     fn bias_mut(&mut self) -> &mut Array<T, Self::Dim> {
//         self.bias.as_mut().unwrap()
//     }

//     fn set_bias(&mut self, bias: Array<T, Self::Dim>) {
//         self.bias = Some(bias);
//     }
// }

impl<T, D> Weighted<T> for LinearParams<T, D>
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

impl<A, B, T, D> Predict<A> for LinearParams<T, D>
where
    A: Dot<Array<T, D>, Output = B>,
    B: for<'a> ops::Add<&'a Array<T, D::Smaller>, Output = B>,
    D: RemoveAxis,
    T: NdFloat,
{
    type Output = B;

    fn predict(&self, input: &A) -> Result<Self::Output, PredictError> {
        let wt = self.weights().t().to_owned();
        let res = input.dot(&wt);
        if let Some(bias) = self.bias() {
            return Ok(res + bias);
        }
        Ok(res)
    }
}

impl<'a, A, B, T, D> Predict<A> for &'a LinearParams<T, D>
where
    A: Dot<Array<T, D>, Output = B>,
    B: ops::Add<&'a Array<T, D::Smaller>, Output = B>,
    D: RemoveAxis,
    T: NdFloat,
{
    type Output = B;

    fn predict(&self, input: &A) -> Result<Self::Output, PredictError> {
        let wt = self.weights().t().to_owned();
        let res = input.dot(&wt);
        if let Some(bias) = self.bias() {
            return Ok(res + bias);
        }
        Ok(res)
    }
}

impl<T> IntoIterator for LinearParams<T>
where
    T: Float,
{
    type Item = Node<T>;
    type IntoIter = vec::IntoIter<Self::Item>;

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

impl<T> FromIterator<Node<T>> for LinearParams<T>
where
    T: Float,
{
    fn from_iter<I: IntoIterator<Item = Node<T>>>(nodes: I) -> Self {
        let nodes = nodes.into_iter().collect::<Vec<_>>();
        let mut iter = nodes.iter();
        let node = iter.next().unwrap();
        let shape = Features::new(node.features(), nodes.len());
        let mut params = Self::zeros(shape);
        params.set_node(0, node.clone());
        for (i, node) in iter.into_iter().enumerate() {
            params.set_node(i + 1, node.clone());
        }
        params
    }
}

#[cfg(feature = "serde")]

mod impl_serde {
    use super::*;
    use serde::{Deserialize, Serialize};
    impl<'a, T, D> Deserialize<'a> for LinearParams<T, D>
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

    impl<T, D> Serialize for LinearParams<T, D>
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
