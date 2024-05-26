/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Perceptron;

pub struct Mlp<I, H, O> {
    input: Perceptron<I, H>,
    hidden: H,
    output: O,
}