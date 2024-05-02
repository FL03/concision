/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::model::Features;
use crate::Node;
use core::ops;
use nd::linalg::Dot;
use nd::*;
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
    pub(crate) bias: Option<Array<T, D::Smaller>>,
    pub(crate) features: D,
    pub(crate) weights: Array<T, D>,
}

impl<A, D> LinearParams<A, D>
where
    D: RemoveAxis,
{
    pub fn new(biased: bool, dim: impl IntoDimension<Dim = D>) -> Self
    where
        A: Clone + Default,
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
        A: Clone + One,
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
        A: Clone + Zero,
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

    pub fn activate<F>(&mut self, f: F) -> LinearParams<A, D>
    where
        F: for<'a> Fn(&'a A) -> A,
    {
        LinearParams {
            bias: self.bias().map(|b| b.map(|b| f(b))),
            features: self.features.clone(),
            weights: self.weights().map(|w| f(w)),
        }
    }

    pub fn bias(&self) -> Option<&Array<A, D::Smaller>> {
        self.bias.as_ref()
    }

    pub fn bias_mut(&mut self) -> Option<&mut Array<A, D::Smaller>> {
        self.bias.as_mut()
    }

    pub fn unbiased(self) -> Self {
        Self { bias: None, ..self }
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

    pub fn linear<T, B>(&self, data: &T) -> B
    where
        T: Dot<Array<A, D>, Output = B>,
        B: for<'a> ops::Add<&'a Array<A, D::Smaller>, Output = B>,
        A: NdFloat,
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

    pub fn weights(&self) -> &Array<A, D> {
        &self.weights
    }

    pub fn weights_mut(&mut self) -> &mut Array<A, D> {
        &mut self.weights
    }
}

impl<T> LinearParams<T> {
    pub fn set_node(&mut self, idx: usize, node: Node<T>)
    where
        T: Float,
    {
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
impl<'a, T, D> serde::Deserialize<'a> for LinearParams<T, D>
where
    T: serde::Deserialize<'a>,
    D: serde::Deserialize<'a> + nd::RemoveAxis,
    <D as nd::Dimension>::Smaller: serde::Deserialize<'a> + nd::Dimension,
{
    fn deserialize<Der>(deserializer: Der) -> Result<Self, Der::Error>
    where
        Der: serde::Deserializer<'a>,
    {
        let (bias, features, weights) = serde::Deserialize::deserialize(deserializer)?;
        Ok(Self {
            bias,
            features,
            weights,
        })
    }
}
#[cfg(feature = "serde")]

impl<T, D> serde::Serialize for LinearParams<T, D>
where
    T: serde::Serialize,
    D: nd::RemoveAxis + serde::Serialize,
    <D as nd::Dimension>::Smaller: nd::Dimension + serde::Serialize,
{
    fn serialize<Ser>(&self, serializer: Ser) -> Result<Ser::Ok, Ser::Error>
    where
        Ser: serde::Serializer,
    {
        (self.bias(), self.features(), self.weights()).serialize(serializer)
    }
}
