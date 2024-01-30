/*
    Appellation: stack <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{Features, Forward, LayerParams, LayerShape};
use ndarray::prelude::{Array2, Dimension, Ix2, NdFloat};
use ndarray::IntoDimension;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num::Float;

use serde::{Deserialize, Serialize};
use std::ops;

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct ModelParams<T = f64>
where
    T: Float,
{
    children: Vec<LayerParams<T>>,
}

impl<T> ModelParams<T>
where
    T: Float,
{
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
        }
    }

    pub fn with_capacity(depth: usize) -> Self {
        Self {
            children: Vec::with_capacity(depth),
        }
    }

    pub fn with_shapes<Sh>(shapes: impl IntoIterator<Item = Sh>) -> Self
    where
        Sh: IntoDimension<Dim = Ix2>,
    {
        let tmp = Vec::from_iter(shapes.into_iter().map(IntoDimension::into_dimension));
        let mut children = Vec::new();
        for (inputs, outputs) in tmp.iter().map(|s| s.into_pattern()) {
            let features = LayerShape::new(inputs, outputs);
            children.push(LayerParams::new(features));
        }
        Self { children }
    }

    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    pub fn build_layers(mut self, shapes: impl IntoIterator<Item = (usize, usize)>) -> Self {
        // let shapes = shapes.into_iter().map(|s| (s.inputs(), s.outputs()));
        for (inputs, outputs) in shapes.into_iter() {
            let features = LayerShape::new(inputs, outputs);
            self.children.push(LayerParams::new(features));
        }
        self
    }

    pub fn len(&self) -> usize {
        self.children.len()
    }

    pub fn params(&self) -> &[LayerParams<T>] {
        &self.children
    }

    pub fn params_mut(&mut self) -> &mut [LayerParams<T>] {
        &mut self.children
    }

    pub fn pop(&mut self) -> Option<LayerParams<T>> {
        self.children.pop()
    }

    pub fn push(&mut self, params: LayerParams<T>) {
        self.children.push(params);
    }

    pub fn validate_shapes(&self) -> bool {
        let mut dim = true;
        for (i, layer) in self.children[..(self.len() - 1)].iter().enumerate() {
            dim = dim && layer.features().outputs() == self.children[i + 1].features().inputs();
        }
        dim
    }
}

impl<T> ModelParams<T>
where
    T: Float + SampleUniform,
    StandardNormal: Distribution<T>,
{
    pub fn init(mut self, biased: bool) -> Self {
        self.children
            .iter_mut()
            .for_each(|l| *l = l.clone().init(biased));
        self
    }

    pub fn init_bias(mut self) -> Self {
        self.children
            .iter_mut()
            .for_each(|l| *l = l.clone().init_bias());
        self
    }

    pub fn init_weight(mut self) -> Self {
        self.children
            .iter_mut()
            .for_each(|l| *l = l.clone().init_weight());
        self
    }
}

impl<T> ModelParams<T> where T: NdFloat {}

impl<T> Forward<Array2<T>> for ModelParams<T>
where
    T: NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, input: &Array2<T>) -> Array2<T> {
        let mut iter = self.children.iter();

        let mut output = iter.next().unwrap().forward(input);
        for layer in iter {
            output = layer.forward(&output);
        }
        output
    }
}

impl<T> AsRef<[LayerParams<T>]> for ModelParams<T>
where
    T: Float,
{
    fn as_ref(&self) -> &[LayerParams<T>] {
        &self.children
    }
}

impl<T> AsMut<[LayerParams<T>]> for ModelParams<T>
where
    T: Float,
{
    fn as_mut(&mut self) -> &mut [LayerParams<T>] {
        &mut self.children
    }
}

impl<T> FromIterator<LayerShape> for ModelParams<T>
where
    T: Float,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = LayerShape>,
    {
        Self {
            children: iter.into_iter().map(LayerParams::new).collect(),
        }
    }
}

impl<T> FromIterator<LayerParams<T>> for ModelParams<T>
where
    T: Float,
{
    fn from_iter<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = LayerParams<T>>,
    {
        Self {
            children: iter.into_iter().collect(),
        }
    }
}

impl<T> IntoIterator for ModelParams<T>
where
    T: Float,
{
    type Item = LayerParams<T>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.children.into_iter()
    }
}

impl<T> ops::Index<usize> for ModelParams<T>
where
    T: Float,
{
    type Output = LayerParams<T>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.children[index]
    }
}

impl<T> ops::IndexMut<usize> for ModelParams<T>
where
    T: Float,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.children[index]
    }
}

impl<T> ops::Index<ops::Range<usize>> for ModelParams<T>
where
    T: Float,
{
    type Output = [LayerParams<T>];

    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        &self.children[index]
    }
}

impl<T> ops::IndexMut<ops::Range<usize>> for ModelParams<T>
where
    T: Float,
{
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut Self::Output {
        &mut self.children[index]
    }
}

impl<T> ops::Index<ops::RangeFrom<usize>> for ModelParams<T>
where
    T: Float,
{
    type Output = [LayerParams<T>];

    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        &self.children[index]
    }
}

impl<T> ops::IndexMut<ops::RangeFrom<usize>> for ModelParams<T>
where
    T: Float,
{
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut Self::Output {
        &mut self.children[index]
    }
}

impl<T> ops::Index<ops::RangeFull> for ModelParams<T>
where
    T: Float,
{
    type Output = [LayerParams<T>];

    fn index(&self, index: ops::RangeFull) -> &Self::Output {
        &self.children[index]
    }
}

impl<T> ops::IndexMut<ops::RangeFull> for ModelParams<T>
where
    T: Float,
{
    fn index_mut(&mut self, index: ops::RangeFull) -> &mut Self::Output {
        &mut self.children[index]
    }
}

impl<T> ops::Index<ops::RangeInclusive<usize>> for ModelParams<T>
where
    T: Float,
{
    type Output = [LayerParams<T>];

    fn index(&self, index: ops::RangeInclusive<usize>) -> &Self::Output {
        &self.children[index]
    }
}

impl<T> ops::IndexMut<ops::RangeInclusive<usize>> for ModelParams<T>
where
    T: Float,
{
    fn index_mut(&mut self, index: ops::RangeInclusive<usize>) -> &mut Self::Output {
        &mut self.children[index]
    }
}

impl<T> ops::Index<ops::RangeTo<usize>> for ModelParams<T>
where
    T: Float,
{
    type Output = [LayerParams<T>];

    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        &self.children[index]
    }
}

impl<T> ops::IndexMut<ops::RangeTo<usize>> for ModelParams<T>
where
    T: Float,
{
    fn index_mut(&mut self, index: ops::RangeTo<usize>) -> &mut Self::Output {
        &mut self.children[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::LayerShape;

    #[test]
    fn test_model_params() {
        let (inputs, outputs) = (5, 3);

        let shapes = [(inputs, outputs), (outputs, outputs), (outputs, 1)];

        let params = ModelParams::<f64>::new().build_layers(shapes).init(true);

        // validate the dimensions of the model params
        assert!(params.validate_shapes());

        for (layer, shape) in params.into_iter().zip(&shapes) {
            assert_eq!(layer.features(), &LayerShape::new(shape.0, shape.1));
        }
    }
}
