/*
   Appellation: base <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Dimension};

pub trait Apply<T> {
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T;
}

impl<T, D> Apply<T> for Array<T, D>
where
    D: Dimension,
{
    fn apply<F>(&self, f: F) -> Self
    where
        F: Fn(&T) -> T,
    {
        self.map(f)
    }
}

pub trait ApplyTo<T> {
    fn apply_to(&self, args: &mut T) -> &mut T;
}

pub trait As<T>: AsRef<T> + AsMut<T> {}

impl<S, T> As<T> for S where S: AsRef<T> + AsMut<T> {}
