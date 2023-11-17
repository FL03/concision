/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Array1, Array2, Dimension};
use num::Float;

pub trait Gradient<T = f64> where T: Float {
    fn partial(&self, x: T) -> T;

    fn gradient<D>(&self, args: &Array<T, D>) -> Array<T, D> where D: Dimension {
        args.mapv(|xs| self.partial(xs))
    }
}

pub trait Objective<T> {
    type Model;

    fn objective(&self, x: &Array2<T>, y: &Array1<T>) -> Array1<T>;
}

pub trait PartialDerivative<T> {
    type Args;

    fn partial_derivative(&self, args: Self::Args) -> T;
}

pub trait Minimize<T> {
    fn minimize(&self, scale: T) -> Self;
}


pub trait LearningRate<T = f64>
where
    T: Float,
{
    fn gamma(&self) -> T;
}

pub trait Momentum<T = f64>
where
    T: Float,
{
    fn mu(&self) -> T; // Momentum Rate

    fn nestrov(&self) -> bool;
}

pub trait Decay<T = f64>
where
    T: Float,
{
    fn lambda(&self) -> T; // Decay Rate
}

pub trait Dampener<T = f64>
where
    T: Float,
{
    fn tau(&self) -> T; // Momentum Damper
}
