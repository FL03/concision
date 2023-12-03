/*
    Appellation: stack <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{Features, LayerParams, LayerPosition, LayerShape};
use ndarray::prelude::{Dimension, Ix2};
use ndarray::IntoDimension;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use num::Float;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops;

pub struct ModelMap<T = f64>
where
    T: Float,
{
    store: HashMap<LayerPosition, LayerParams<T>>,
}

impl<T> ModelMap<T>
where
    T: Float,
{
    pub fn with_shapes<Sh>(shapes: impl IntoIterator<Item = Sh>) -> Self
    where
        Sh: IntoDimension<Dim = Ix2>,
    {
        let tmp = Vec::from_iter(shapes.into_iter().map(IntoDimension::into_dimension));
        let mut store = HashMap::new();
        for (i, (inputs, outputs)) in tmp.iter().map(|s| s.into_pattern()).enumerate() {
            let features = LayerShape::new(inputs, outputs);
            let position = if i == 0 {
                LayerPosition::input()
            } else if i == tmp.len() - 1 {
                LayerPosition::output(i)
            } else {
                LayerPosition::hidden(i)
            };
            store.insert(position, LayerParams::new(features));
        }
        Self { store }
    }
}

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
        for (i, (inputs, outputs)) in tmp.iter().map(|s| s.into_pattern()).enumerate() {
            let features = LayerShape::new(inputs, outputs);
            let position = if i == 0 {
                LayerPosition::input()
            } else if i == tmp.len() - 1 {
                LayerPosition::output(i)
            } else {
                LayerPosition::hidden(i)
            };

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
{
    pub fn init_layers(mut self, biased: bool) -> Self {
        self.children
            .iter_mut()
            .for_each(|l| *l = l.clone().init(biased));
        self
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

        let params = ModelParams::<f64>::new()
            .build_layers(shapes)
            .init_layers(true);

        // validate the dimensions of the model params
        assert!(params.validate_shapes());

        for (layer, shape) in params.into_iter().zip(&shapes) {
            assert_eq!(layer.features(), &LayerShape::new(shape.0, shape.1));
        }
    }
}
