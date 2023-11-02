/*
    Appellation: network <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::layers::Layer;
use num::Float;

pub struct NeuralNetwork<T: Float = f64> {
    pub layers: Vec<Layer<T>>,
}
