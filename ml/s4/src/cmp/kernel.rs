/*
    Appellation: kernel <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array2;
use num::Float;

pub struct Kernel<T = f64> {
    kernal: Array2<T>,
}

impl<T> Kernel<T>
where
    T: Float,
{
    pub fn new(kernal: Array2<T>) -> Self {
        Self { kernal }
    }

    pub fn square(features: usize) -> Self
    where
        T: Default,
    {
        let kernal = Array2::<T>::default((features, features));
        Self::new(kernal)
    }

    pub fn kernal(&self) -> &Array2<T> {
        &self.kernal
    }
}