/*
    Appellation: norm <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # norm
//!
pub use self::{kinds::*, utils::*};

pub(crate) mod kinds;

use ndarray::prelude::{Array, NdFloat};
use ndarray::Dimension;

pub trait Normalize<T> {
    fn l0(&self) -> T;

    fn l1(&self) -> T;

    fn l2(&self) -> T;
}

impl<T, D> Normalize<T> for Array<T, D>
where
    D: Dimension,
    T: NdFloat,
{
    fn l0(&self) -> T {
        utils::l0_norm(self)
    }

    fn l1(&self) -> T {
        utils::l1_norm(self)
    }

    fn l2(&self) -> T {
        utils::l2_norm(self)
    }
}

pub(crate) mod utils {
    use ndarray::prelude::{Array, NdFloat};
    use ndarray::Dimension;
    use ndarray_stats::QuantileExt;

    pub fn l0_norm<T, D>(args: &Array<T, D>) -> T
    where
        D: Dimension,
        T: NdFloat,
    {
        *args.max().expect("No max value")
    }

    pub fn l1_norm<T, D>(args: &Array<T, D>) -> T
    where
        D: Dimension,
        T: NdFloat,
    {
        args.mapv(|xs| xs.abs()).sum()
    }

    pub fn l2_norm<T, D>(args: &Array<T, D>) -> T
    where
        D: Dimension,
        T: NdFloat,
    {
        args.mapv(|xs| xs.powi(2)).sum().sqrt()
    }
}

#[cfg(test)]
mod tests {}
