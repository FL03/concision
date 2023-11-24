/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension};
use num::Float;

pub trait Gradient<T = f64>
where
    T: Float,
{
    fn partial(&self, x: T) -> T;

    fn gradient<D>(&self, args: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
    {
        args.mapv(|xs| self.partial(xs))
    }
}

pub trait Minimize<T> {
    fn minimize(&self, scale: T) -> Self;
}

pub trait Dampener<T = f64>
where
    T: Float,
{
    fn tau(&self) -> T; // Momentum Damper
}

pub trait Decay<T = f64>
where
    T: Float,
{
    fn lambda(&self) -> T; // Decay Rate
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
