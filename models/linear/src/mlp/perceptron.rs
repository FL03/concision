/*
   Appellation: perceptron <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Biased, Linear};
use concision::prelude::{relu, Activate, Predict, PredictError};
use nd::prelude::*;
use nd::Data;
use num::traits::Zero;
pub struct ReLU;

impl<A, S, D> Activate<ArrayBase<S, D>> for ReLU
where
    A: Clone + PartialOrd + Zero,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn activate(&self, args: ArrayBase<S, D>) -> Self::Output {
        args.mapv(relu)
    }
}

impl<'a, A, S, D> Activate<&'a ArrayBase<S, D>> for ReLU
where
    A: Clone + PartialOrd + Zero,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn activate(&self, args: &'a ArrayBase<S, D>) -> Self::Output {
        args.mapv(relu)
    }
}

pub struct Perceptron<A = f64, K = Biased, D = Ix2, F = ReLU>
where
    D: Dimension,
{
    module: Linear<A, K, D>,
    rho: F,
}

impl<A, K, D, F> Perceptron<A, K, D, F>
where
    D: Dimension,
{
    pub fn new(module: Linear<A, K, D>, rho: F) -> Self {
        Self { module, rho }
    }

    pub fn activate<T>(&self, args: T) -> F::Output
    where
        F: Activate<T>,
    {
        self.rho.activate(args)
    }
}

impl<X, Y, Z, A, K, D, F> Predict<X> for Perceptron<A, K, D, F>
where
    D: Dimension,
    F: for<'a> Activate<&'a Y, Output = Z>,
    Linear<A, K, D>: Predict<X, Output = Y>,
{
    type Output = Z;

    fn predict(&self, args: &X) -> Result<Self::Output, PredictError> {
        let res = self.module.predict(args)?;
        Ok(self.rho.activate(&res))
    }
}
