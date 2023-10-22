/*
    Appellation: binary <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::neurons::activate::Activator;
use ndarray::Array;
use num::{One, Zero};

pub struct Heavyside;

impl Heavyside {
    pub fn heavyside<T: PartialOrd + One + Zero>(x: T) -> T {
        if x > T::zero() {
            T::one()
        } else {
            T::zero()
        }
    }
}

impl<T, D> Activator<Array<T, D>> for Heavyside
where
    D: ndarray::Dimension,
    T: Clone + PartialOrd + One + Zero,
{
    fn rho(x: Array<T, D>) -> Array<T, D> {
        x.mapv(|x| Self::heavyside(x))
    }
}
