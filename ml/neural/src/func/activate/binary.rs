/*
    Appellation: binary <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Activate;
use ndarray::prelude::{Array, Dimension};
use num::{One, Zero};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct Heavyside;

impl Heavyside {
    pub fn heavyside<T>(x: T) -> T
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

impl<T, D> Activate<T, D> for Heavyside
where
    D: Dimension,
    T: Clone + One + PartialOrd + Zero,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
        args.mapv(|x| Self::heavyside(x))
    }
}
