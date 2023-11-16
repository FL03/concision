/*
    Appellation: network <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::layers::Layer;
use num::Float;

pub struct NeuralNetwork<T: Float = f64> {
    pub input: Layer<T>,
    pub hidden: Vec<Layer<T>>,
    pub output: Layer<T>,
}
