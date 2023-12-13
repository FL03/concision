/*
    Appellation: sublayers <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::Activate;
use crate::layers::Layer;

use ndarray::prelude::Array2;
use num::Float;

pub trait Wrap<T> {
    type Output;

    fn wrap(&self, obj: T) -> Self::Output;
}

pub trait Wrapper<T = f64>
where
    T: Float,
{
    fn apply(&self, data: &Array2<T>) -> Array2<T>;

    fn wrap<A>(&self, layer: Layer<T, A>)
    where
        A: Activate<T>;

    fn wrapper(&self) -> &Self;

    fn wrapper_mut(&mut self) -> &mut Self;
}
