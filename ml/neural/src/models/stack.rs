/*
    Appellation: stack <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::layers::Layer;
use crate::prelude::{Activate, LayerShape, LinearActivation, Parameterized};
use ndarray::prelude::Ix2;
use num::Float;
use serde::{Deserialize, Serialize};

/// A [Stack] is a collection of [Layer]s, typically used to construct the hidden
/// layers of a deep neural network.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct Stack<T = f64, A = LinearActivation>
where
    A: Activate<T, Ix2>,
    T: Float,
{
    children: Vec<Layer<T, A>>,
}

impl<T, A> Stack<T, A>
where
    A: Activate<T, Ix2> + Default,
    T: Float,
{
    pub fn build_layers(mut self, shapes: impl IntoIterator<Item = (usize, usize)>) -> Self {
        for (inputs, outputs) in shapes.into_iter() {
            self.children
                .push(Layer::<T, A>::from(LayerShape::new(inputs, outputs)));
        }
        self
    }
}

impl<T, A> Stack<T, A>
where
    A: Activate<T, Ix2> + Clone + Default,
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
    A: Activate<T, Ix2>,
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

    pub fn layers(&self) -> &[Layer<T, A>] {
        &self.children
    }

    pub fn layers_mut(&mut self) -> &mut [Layer<T, A>] {
        &mut self.children
    }

    pub fn depth(&self) -> usize {
        self.children.len()
    }

    pub fn validate_shapes(&self) -> bool {
        let mut dim = true;
        for (i, layer) in self.children[..(self.depth() - 1)].iter().enumerate() {
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
    A: Activate<T, Ix2>,
    T: Float,
{
    fn as_ref(&self) -> &[Layer<T, A>] {
        &self.children
    }
}

impl<T, A> AsMut<[Layer<T, A>]> for Stack<T, A>
where
    A: Activate<T, Ix2>,
    T: Float,
{
    fn as_mut(&mut self) -> &mut [Layer<T, A>] {
        &mut self.children
    }
}

impl<T, A> Default for Stack<T, A>
where
    A: Activate<T, Ix2>,
    T: Float,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T, A> IntoIterator for Stack<T, A>
where
    A: Activate<T, Ix2>,
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
    A: Activate<T, Ix2>,
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
    A: Activate<T, Ix2>,
    T: Float,
{
    fn from(layers: Vec<Layer<T, A>>) -> Self {
        Self { children: layers }
    }
}

impl<T, A> From<Layer<T, A>> for Stack<T, A>
where
    A: Activate<T, Ix2>,
    T: Float,
{
    fn from(layer: Layer<T, A>) -> Self {
        Self {
            children: vec![layer],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{LayerShape, Softmax};

    #[test]
    fn test_stack() {
        let (samples, inputs, outputs) = (20, 5, 3);
        let features = LayerShape::new(inputs, outputs);

        let shapes = [(inputs, outputs), (outputs, outputs), (outputs, 1)];

        let stack = Stack::<f64, Softmax>::new()
            .build_layers(shapes)
            .init_layers(true);

        assert!(stack.validate_shapes());
        for (layer, shape) in stack.layers().iter().zip(&shapes) {
            assert_eq!(layer.features(), &LayerShape::new(shape.0, shape.1));
        }
    }
}
