/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array1, Array2};

pub trait Gradient<T> {
    type Model;

    fn gradient(&self, x: &Array2<T>, y: &Array1<T>) -> T;
}

pub trait Objective<T> {
    type Model;

    fn objective(&self, x: &Array2<T>, y: &Array1<T>) -> Array1<T>;
}

pub trait PartialDerivative<T> {
    type Model;

    fn partial_derivative(&self, x: &Array2<T>, y: &Array1<T>) -> Array2<T>;
}

pub trait Optimize {
    fn optimize(&self, params: &mut dyn Optimizable);
}

pub trait Optimizable {}
