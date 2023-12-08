/*
    Appellation: group <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Biased, Weighted};
use crate::core::prelude::GenerateRandom;
use crate::prelude::{Forward, Node};
use ndarray::prelude::{s, Array, Axis, Dimension, Ix1, Ix2};
use ndarray::{IntoDimension, RemoveAxis};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ParamGroup<T = f64, D = Ix2>
where
    T: Float,
    D: Dimension,
{
    bias: Array<T, D::Smaller>,
    features: D,
    weights: Array<T, D>,
}

impl<T, D> ParamGroup<T, D>
where
    T: Float,
    D: Dimension + RemoveAxis,
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

    pub fn features_mut(&mut self) -> &mut D {
        &mut self.features
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

    // pub fn set_node(&mut self, idx: usize, node: ParamGroup<T, D::Smaller>) {
    //     let dim = self.features();

    //     self.weights
    //                 .slice_mut(s![idx, ..])
    //         .assign(&node.weights);
    //     self.bias.slice_mut(s![idx, ..]).assign(&node.bias);
    // }
}

impl<T, D> ParamGroup<T, D>
where
    T: Float + SampleUniform,
    D: Dimension + RemoveAxis,
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

impl<T, D> Biased<T, D> for ParamGroup<T, D>
where
    T: Float,
    D: Dimension + RemoveAxis,
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
    D: Dimension,
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

impl<T> Forward<Array<T, Ix2>> for ParamGroup<T, Ix1>
where
    T: Float + 'static,
{
    type Output = Array<T, Ix1>;

    fn forward(&self, data: &Array<T, Ix2>) -> Self::Output {
        data.dot(self.weights()) + self.bias()
    }
}

impl<T> Forward<Array<T, Ix2>> for ParamGroup<T, Ix2>
where
    T: Float + 'static,
{
    type Output = Array<T, Ix2>;

    fn forward(&self, data: &Array<T, Ix2>) -> Self::Output {
        data.dot(self.weights()) + self.bias()
    }
}

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

// impl<T> IntoIterator for ParamGroup<T, Ix2>
// where
//     T: Float,
// {
//     type Item = Node<T>;
//     type IntoIter = std::vec::IntoIter<Self::Item>;

//     fn into_iter(self) -> Self::IntoIter {
//         self.weights()
//             .axis_iter(Axis(0))
//             .zip(self.bias().axis_iter(Axis(0)))
//             .map(|(w, b)| (w.to_owned(), b.to_owned()).into())
//             .collect::<Vec<_>>()
//             .into_iter()
//     }
// }

impl<T, D> IntoIterator for ParamGroup<T, D>
where
    D: Dimension + RemoveAxis,
    T: Float,
    <D as Dimension>::Smaller: Dimension + RemoveAxis,
{
    type Item = (
        Array<T, D::Smaller>,
        Array<T, <D::Smaller as Dimension>::Smaller>,
    );
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.weights()
            .axis_iter(Axis(0))
            .zip(self.bias().axis_iter(Axis(0)))
            .map(|(w, b)| (w.to_owned(), b.to_owned()))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

// impl<T, D> FromIterator<ParamGroup<T, D::Smaller>> for ParamGroup<T, D>
// where
//     T: Float,
//     D: Dimension + RemoveAxis,
//     <D as Dimension>::Smaller: Dimension + RemoveAxis,
// {
//     fn from_iter<I>(iter: I) -> Self
//     where
//         I: IntoIterator<Item = ParamGroup<T, D::Smaller>>,
//     {
//         let store = Vec::from_iter(iter);
//         let mut features = vec![store.len()];
//         features.extend(store.first().unwrap().weights().shape());

//         let mut group = Self::new(features.as_slice());
//         let mut iter = iter.into_iter();
//         let mut group = Self::new(weights.shape());
//         group.set_weights(weights);
//         group.set_bias(bias);
//         group
//     }
// }
