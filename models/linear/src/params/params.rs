/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::build_bias;
use crate::model::Features;
use core::ops;
use nd::linalg::Dot;
use nd::*;
use num::{Float, One, Zero};

#[cfg(no_std)]
use alloc::vec;
#[cfg(feature = "std")]
use std::vec;

pub(crate) type Node<T> = (Array<T, Ix1>, Option<Array<T, Ix0>>);

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

    pub fn default(dim: impl IntoDimension<Dim = D>) -> Self
    where
        A: Clone + Default,
    {
        let dim = dim.into_dimension();
        let bias = build_bias(true, dim.clone(), |dim| Array::default(dim));
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
        T: Clone + Default,
    {
        let (weight, bias) = node;
        if let Some(bias) = bias {
            if !self.is_biased() {
                let mut tmp = Array1::default(self.outputs());
                tmp.index_axis_mut(Axis(0), idx).assign(&bias);
                self.bias = Some(tmp);
            }
            self.bias
                .as_mut()
                .unwrap()
                .index_axis_mut(Axis(0), idx)
                .assign(&bias);
        }

        self.weights_mut()
            .index_axis_mut(Axis(0), idx)
            .assign(&weight);
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
                .map(|(w, b)| (w.to_owned(), Some(b.to_owned())))
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

impl<T> FromIterator<(Array1<T>, Option<Array0<T>>)> for LinearParams<T, Ix2>
where
    T: Clone + Default,
{
    fn from_iter<I: IntoIterator<Item = (Array1<T>, Option<Array0<T>>)>>(nodes: I) -> Self {
        let nodes = nodes.into_iter().collect::<Vec<_>>();
        let mut iter = nodes.iter();
        let node = iter.next().unwrap();
        let shape = Features::new(node.0.shape()[0], nodes.len());
        let mut params = LinearParams::default(shape);
        params.set_node(0, node.clone());
        for (i, node) in iter.into_iter().enumerate() {
            params.set_node(i + 1, node.clone());
        }
        params
    }
}

impl<A> From<(Array<A, Ix1>, A)> for LinearParams<A, Ix1>
where
    A: Clone,
{
    fn from((weights, bias): (Array<A, Ix1>, A)) -> Self {
        let bias = Array::from_elem((), bias);
        Self {
            bias: Some(bias),
            features: weights.raw_dim(),
            weights,
        }
    }
}

impl<A> From<(Array<A, Ix1>, Option<A>)> for LinearParams<A, Ix1>
where
    A: Clone,
{
    fn from((weights, bias): (Array<A, Ix1>, Option<A>)) -> Self {
        let bias = bias.map(|b| Array::from_elem((), b));
        Self {
            bias,
            features: weights.raw_dim(),
            weights,
        }
    }
}
impl<T, D> From<(Array<T, D>, Array<T, D::Smaller>)> for LinearParams<T, D>
where
    D: RemoveAxis,
{
    fn from((weights, bias): (Array<T, D>, Array<T, D::Smaller>)) -> Self {
        Self {
            bias: Some(bias),
            features: weights.raw_dim(),
            weights,
        }
    }
}

impl<T, D> From<(Array<T, D>, Option<Array<T, D::Smaller>>)> for LinearParams<T, D>
where
    D: RemoveAxis,
{
    fn from((weights, bias): (Array<T, D>, Option<Array<T, D::Smaller>>)) -> Self {
        Self {
            bias,
            features: weights.raw_dim(),
            weights,
        }
    }
}
