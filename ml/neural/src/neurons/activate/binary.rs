/*
    Appellation: binary <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neurons::activate::Activate;
use ndarray::prelude::Array;
use ndarray::Dimension;
use num::{One, Zero};

pub struct Heavyside;

impl Heavyside {
    pub fn heavyside<T: PartialOrd + One + Zero>(x: T) -> T
    where
        T: One + PartialOrd + Zero,
    {
        if x > T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }
}

impl<T, D> Activate<Array<T, D>> for Heavyside
where
    D: Dimension,
    T: Clone + One + PartialOrd + Zero,
{
    fn activate(&self, x: Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::heavyside(x))
    }
}
