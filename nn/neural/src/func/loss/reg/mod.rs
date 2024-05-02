/*
    Appellation: reg <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{huber::*, mae::*, mse::*, utils::*};

pub(crate) mod huber;
pub(crate) mod mae;
pub(crate) mod mse;

use crate::func::loss::Loss;
use ndarray::prelude::{Array, Dimension, NdFloat};
use num::FromPrimitive;

pub enum RegressiveLoss {
    Huber(HuberLoss),
    MeanAbsoluteError,
    MeanSquaredError,
    Other(String),
}

pub trait RegressiveLosses<T = f64> {
    fn huber(&self, delta: T, other: &Self) -> T;
    fn mae(&self, other: &Self) -> T;
    fn mse(&self, other: &Self) -> T;
}

impl<T, D> RegressiveLosses<T> for Array<T, D>
where
    D: Dimension,
    T: FromPrimitive + NdFloat,
{
    fn huber(&self, delta: T, other: &Self) -> T {
        HuberLoss::new(delta).loss(self, other)
    }

    fn mae(&self, other: &Self) -> T {
        MeanAbsoluteError::new().loss(self, other)
    }

    fn mse(&self, other: &Self) -> T {
        MeanSquaredError::new().loss(self, other)
    }
}

pub(crate) mod utils {
    use ndarray::prelude::{Array, Dimension, NdFloat};
    use num::FromPrimitive;

    pub fn mae<T, D>(pred: &Array<T, D>, target: &Array<T, D>) -> Option<T>
    where
        D: Dimension,
        T: FromPrimitive + NdFloat,
    {
        (pred - target).mapv(T::abs).mean()
    }

    pub fn mse<T, D>(pred: &Array<T, D>, target: &Array<T, D>) -> Option<T>
    where
        D: Dimension,
        T: FromPrimitive + NdFloat,
    {
        (pred - target).mapv(|x| x.powi(2)).mean()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{GenerateRandom, RoundTo};
    use crate::func::loss::Loss;
    use ndarray::prelude::{Array, Ix2};
    use ndarray_stats::DeviationExt;

    #[test]
    fn test_mae() {
        let (m, n) = (3, 3);
        let shape = (m, n);

        let ns = m * n;

        let targets: Array<f64, Ix2> = Array::linspace(0.0, ns as f64, ns)
            .into_shape(shape)
            .unwrap();
        let pred = Array::<f64, Ix2>::uniform_between(3.0, shape);

        let loss = MeanAbsoluteError::new().loss(&pred, &targets).round_to(4);

        let exp = targets.mean_abs_err(&pred).unwrap().round_to(4);

        assert_eq!(&loss, &exp);
    }

    #[test]
    fn test_mse() {
        let (m, n) = (3, 3);
        let shape = (m, n);

        let ns = m * n;

        let targets: Array<f64, Ix2> = Array::linspace(0.0, ns as f64, ns)
            .into_shape(shape)
            .unwrap();
        let pred = Array::<f64, Ix2>::uniform_between(3.0, shape);

        let loss = MeanSquaredError::new().loss(&pred, &targets).round_to(4);

        let exp = targets.mean_sq_err(&pred).unwrap().round_to(4);

        assert_eq!(&loss, &exp);
    }
}
