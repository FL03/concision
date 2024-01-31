/*
    Appellation: mlp <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Multi-Layer Perceptron
//!

// use crate::func::activate::{Activate, LinearActivation, ReLU, Softmax};
// use crate::layers::{Layer, LayerShape,};
// use crate::layers::seq::Sequential;

// use crate::prelude::{Features, Forward};

// use ndarray::prelude::{Array2, Ix2, NdFloat};
// use ndarray::IntoDimension;
// use num::Float;

// pub struct MLP<T = f64, I = LinearActivation, O = Softmax>
// where
//     I: Activate<T>,
//     H: Activate<T>,
//     O: Activate<T>,
// {
//     pub input: Layer<T, I>,
//     pub hidden: Sequential<T>,
//     pub output: Layer<T, O>,
// }

// impl<T, I, H, O> MLP<T, I, H, O>
// where
//     T: Float,
//     I: Activate<T>,
//     H: Activate<T>,
//     O: Activate<T>,
// {
//     pub fn new(input: Layer<T, I>, hidden: Stack<T, H>, output: Layer<T, O>) -> Self {
//         Self {
//             input,
//             hidden,
//             output,
//         }
//     }

//     pub fn input(&self) -> &Layer<T, I> {
//         &self.input
//     }

//     pub fn hidden(&self) -> &Stack<T, H> {
//         &self.hidden
//     }

//     pub fn output(&self) -> &Layer<T, O> {
//         &self.output
//     }
// }
// impl<T, I, H, O> MLP<T, I, H, O>
// where
//     T: Default,
//     I: Activate<T> + Default,
//     H: Activate<T> + Default,
//     O: Activate<T> + Default,
// {
//     pub fn create<Sh>(hidden: usize, inputs: usize, outputs: usize) -> Self
//     where
//         Sh: IntoDimension<Dim = Ix2>,
//     {
//         let input_shape = LayerShape::new(inputs, outputs);
//         let shape = LayerShape::new(outputs, outputs);

//         let input = Layer::from(input_shape);
//         let stack = Stack::square(hidden, outputs, outputs);
//         let output = Layer::from(shape);
//         Self::new(input, stack, output)
//     }

//     pub fn from_features(
//         inputs: (usize, usize),
//         hidden: impl IntoIterator<Item = (usize, usize)>,
//         outputs: (usize, usize),
//     ) -> Self {
//         let input = Layer::from(LayerShape::from(inputs));
//         let stack = Stack::new().build_layers(hidden);
//         let output = Layer::from(LayerShape::from(outputs));
//         Self::new(input, stack, output)
//     }

//     // pub fn from_shapes<Sh>(inputs: Sh, hidden: impl IntoIterator<Item = Sh>, outputs: Sh) -> Self where Sh: IntoDimension<Dim = Ix2> {
//     //     let input = LayerShape::from_dimension(inputs);
//     //     let hidden = Stack::new().build_layers(hidden);
//     //     let output = LayerShape::from_dimension(outputs);

//     //     Self::new(Layer::from(input), hidden, Layer::new(output))
//     // }
// }

// impl<T, I, H, O> MLP<T, I, H, O>
// where
//     T: Float,
//     I: Activate<T> + Clone,
//     H: Activate<T> + Clone,
//     O: Activate<T> + Clone,
// {
//     pub fn validate_dims(&self) -> bool {
//         self.hidden.validate_shapes()
//             && self.input.features().outputs() == self.hidden.first().unwrap().features().inputs()
//             && self.output.features().inputs() == self.hidden.last().unwrap().features().outputs()
//     }
// }

// impl<T, I, H, O> Forward<Array2<T>> for MLP<T, I, H, O>
// where
//     T: NdFloat,
//     I: Activate<T> + Clone,
//     H: Activate<T> + Clone,
//     O: Activate<T> + Clone,
// {
//     type Output = Array2<T>;

//     fn forward(&self, args: &Array2<T>) -> Self::Output {
//         let mut out = self.input.forward(args);
//         for layer in self.hidden.clone().into_iter() {
//             out = layer.forward(&out);
//         }
//         self.output.forward(&out)
//     }
// }
