/*
    Appellation: stack <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::layers::{Layer, LayerShape};
use crate::prelude::{Activate, Features, Linear, Parameterized};
// use ndarray::{IntoDimension, Ix2};
use num::Float;
use serde::{Deserialize, Serialize};
use std::ops;

pub trait Layers<T, A>: IntoIterator<Item = Layer<T, A>>
where
    A: Activate<T>,
    T: Float,
{
}

pub trait StackExt<T, A>
where
    A: Activate<T>,
    T: Float,
    Self: Clone + Layers<T, A> + IntoIterator<Item = Layer<T, A>>,
{
    fn validate_params(&self) -> bool {
        let layers = self.clone().into_iter().collect::<Vec<_>>();
        let depth = layers.len();
        let mut dim = true;
        for (i, layer) in layers[..(depth - 1)].into_iter().enumerate() {
            dim = dim && layer.features().inputs() == layers[i + 1].features().outputs();
        }
        dim
    }
}

/// A [Stack] is a collection of [Layer]s, typically used to construct the hidden
/// layers of a deep neural network.
#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
pub struct Stack<T = f64, A = Linear>
where
    A: Activate<T>,
    T: Float,
{
    children: Vec<Layer<T, A>>,
}

impl<T, A> Stack<T, A>
where
    A: Activate<T> + Default,
    T: Float,
{
    // pub fn create<Sh: IntoDimension<Dim = Ix2>>(shapes: impl IntoIterator<Item = Sh>) -> Self {
    //     let shapes = shapes.into_iter().map(|s| s.into_dimension());
    //     let mut children = Vec::new();
    //     for (inputs, outputs) in shapes {
    //         children.push(Layer::<T, A>::from(LayerShape::new(inputs, outputs)));
    //     }

    // }
    // pub fn build_layers<Sh: Features>(mut self, shapes: impl IntoIterator<Item = Sh>) -> Self {
    //     // let shapes = shapes.into_iter().map(|s| (s.inputs(), s.outputs()));
    //     for (inputs, outputs) in shapes.into_iter().map(|s| (s.inputs(), s.outputs())) {
    //         self.children
    //             .push(Layer::<T, A>::from(LayerShape::new(inputs, outputs)));
    //     }
    //     self
    // }
    pub fn build_layers(mut self, shapes: impl IntoIterator<Item = (usize, usize)>) -> Self {
        // let shapes = shapes.into_iter().map(|s| (s.inputs(), s.outputs()));
        for (inputs, outputs) in shapes.into_iter() {
            self.children
                .push(Layer::<T, A>::from(LayerShape::new(inputs, outputs)));
        }
        self
    }
}

impl<T, A> Stack<T, A>
where
    A: Activate<T> + Clone + Default,
    T: Float + crate::core::prelude::SampleUniform,
{
    pub fn init_layers(mut self, biased: bool) -> Self {
        self.children
            .iter_mut()
            .for_each(|l| *l = l.clone().init(biased));
        self
    }
}

impl<T, A> Stack<T, A>
where
    A: Activate<T>,
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

    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }

    pub fn first(&self) -> Option<&Layer<T, A>> {
        self.children.first()
    }

    pub fn first_mut(&mut self) -> Option<&mut Layer<T, A>> {
        self.children.first_mut()
    }

    pub fn last(&self) -> Option<&Layer<T, A>> {
        self.children.last()
    }

    pub fn layers(&self) -> &[Layer<T, A>] {
        &self.children
    }

    pub fn layers_mut(&mut self) -> &mut [Layer<T, A>] {
        &mut self.children
    }

    pub fn len(&self) -> usize {
        self.children.len()
    }

    pub fn validate_params(&self) -> bool {
        let mut dim = true;
        for (i, layer) in self[..(self.len() - 1)].iter().enumerate() {
            dim = dim && layer.features().outputs() == self[i + 1].features().inputs();
        }
        dim
    }

    pub fn validate_shapes(&self) -> bool {
        let mut dim = true;
        for (i, layer) in self.children[..(self.len() - 1)].iter().enumerate() {
            dim = dim && layer.features().outputs() == self.children[i + 1].features().inputs();
        }
        dim
    }

    pub fn push(&mut self, layer: Layer<T, A>) {
        self.children.push(layer);
    }

    pub fn pop(&mut self) -> Option<Layer<T, A>> {
        self.children.pop()
    }
}

impl<T, A> AsRef<[Layer<T, A>]> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    fn as_ref(&self) -> &[Layer<T, A>] {
        &self.children
    }
}

impl<T, A> AsMut<[Layer<T, A>]> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    fn as_mut(&mut self) -> &mut [Layer<T, A>] {
        &mut self.children
    }
}

impl<T, A> IntoIterator for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    type Item = Layer<T, A>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.children.into_iter()
    }
}

impl<T, A> FromIterator<Layer<T, A>> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    fn from_iter<I: IntoIterator<Item = Layer<T, A>>>(iter: I) -> Self {
        Self {
            children: iter.into_iter().collect(),
        }
    }
}

impl<T, A> From<Vec<Layer<T, A>>> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    fn from(children: Vec<Layer<T, A>>) -> Self {
        Self { children }
    }
}

impl<T, A> From<Layer<T, A>> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    fn from(layer: Layer<T, A>) -> Self {
        Self {
            children: vec![layer],
        }
    }
}

impl<T, A> ops::Index<usize> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    type Output = Layer<T, A>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.children[index]
    }
}

impl<T, A> ops::IndexMut<usize> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.children[index]
    }
}

impl<T, A> ops::Index<ops::Range<usize>> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    type Output = [Layer<T, A>];

    fn index(&self, index: ops::Range<usize>) -> &Self::Output {
        &self.children[index]
    }
}

impl<T, A> ops::IndexMut<ops::Range<usize>> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    fn index_mut(&mut self, index: ops::Range<usize>) -> &mut Self::Output {
        &mut self.children[index]
    }
}

impl<T, A> ops::Index<ops::RangeFrom<usize>> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    type Output = [Layer<T, A>];

    fn index(&self, index: ops::RangeFrom<usize>) -> &Self::Output {
        &self.children[index]
    }
}

impl<T, A> ops::IndexMut<ops::RangeFrom<usize>> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    fn index_mut(&mut self, index: ops::RangeFrom<usize>) -> &mut Self::Output {
        &mut self.children[index]
    }
}

impl<T, A> ops::Index<ops::RangeTo<usize>> for Stack<T, A>
where
    A: Activate<T>,
    T: Float,
{
    type Output = [Layer<T, A>];

    fn index(&self, index: ops::RangeTo<usize>) -> &Self::Output {
        &self.children[index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{LayerShape, Softmax};

    #[test]
    fn test_stack() {
        let (inputs, outputs) = (5, 3);

        let shapes = [(inputs, outputs), (outputs, outputs), (outputs, 1)];

        let stack = Stack::<f64, Softmax>::new()
            .build_layers(shapes)
            .init_layers(true);
        // assert!(stack.validate_params());

        assert!(stack.validate_shapes());
        for (layer, shape) in stack.layers().iter().zip(&shapes) {
            assert_eq!(layer.features(), &LayerShape::new(shape.0, shape.1));
        }
    }
}
