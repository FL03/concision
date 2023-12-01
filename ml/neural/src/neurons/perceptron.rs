/*
    Appellation: perceptron <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Node;
use crate::prelude::Forward;
use ndarray::prelude::{Array1, Array2, NdFloat};
use num::{Float, FromPrimitive};

pub struct Perceptron<T = f64>
where
    T: Float,
{
    node: Node<T>,
}

impl<T> Perceptron<T>
where
    T: Float,
{
    pub fn new(node: Node<T>) -> Self {
        Self { node }
    }
}

impl<T> Forward<Array2<T>> for Perceptron<T>
where
    T: FromPrimitive + NdFloat,
{
    type Output = Array1<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        self.node.forward(args)
    }
}
