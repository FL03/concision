/*
    Appellation: specs <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait NeuralNetwork: Trainable {

}

pub trait Trainable {
    fn train(&mut self, args: &[f64]) -> f64;
}