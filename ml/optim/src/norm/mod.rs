/*
    Appellation: norm <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # norm
//!
pub use self::{kinds::*, normalizer::*, utils::*};

pub(crate) mod kinds;
pub(crate) mod normalizer;

use ndarray::prelude::{Array, NdFloat};
use ndarray::Dimension;
use ndarray_stats::QuantileExt;

pub trait Normalize<T> {
    type Output;

    fn norm(&self, args: &T) -> Self::Output;
}

pub trait Norm<T> {
    fn l0(&self) -> T;

    fn l1(&self) -> T;

    fn l2(&self) -> T;
}

impl<T, D> Norm<T> for Array<T, D>
where
    D: Dimension,
    T: NdFloat,
{
    fn l0(&self) -> T {
        *self.max().expect("No max value")
    }

    fn l1(&self) -> T {
        self.mapv(|xs| xs.abs()).sum()
    }

    fn l2(&self) -> T {
        self.mapv(|xs| xs.powi(2)).sum().sqrt()
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
mod tests {
    use super::*;

    #[test]
    fn test_l0_norm() {
        let args = Array::linspace(1., 3., 3).into_shape(3).unwrap();

        assert_eq!(l0_norm(&args), 3.);
    }
}
