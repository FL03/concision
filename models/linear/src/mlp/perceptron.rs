/*
   Appellation: perceptron <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::prelude::{Activate, Module, Predict, PredictError, ReLU};
use nd::prelude::*;
use nd::Data;
use num::traits::Zero;

pub struct Rho<T>(T);

impl<T> Rho<T> {
    pub fn new(rho: T) -> Self {
        Self(rho)
    }

    pub fn get(&self) -> &T {
        &self.0
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }

    pub fn activate(&self) -> &T {
        &self.0
    }
}
pub struct Relu;

impl<A, S, D> Activate<ArrayBase<S, D>> for Relu
where
    A: Clone + PartialOrd + Zero,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn activate(&self, args: ArrayBase<S, D>) -> Self::Output {
        args.relu()
    }
}

impl<'a, A, S, D> Activate<&'a ArrayBase<S, D>> for Relu
where
    A: Clone + PartialOrd + Zero,
    D: Dimension,
    S: Data<Elem = A>,
{
    type Output = Array<A, D>;

    fn activate(&self, args: &'a ArrayBase<S, D>) -> Self::Output {
        args.relu()
    }
}

pub struct Perceptron<M, F = Relu>
where
    M: Module,
{
    module: M,
    rho: F,
}

impl<M, F> Perceptron<M, F>
where
    M: Module,
{
    pub fn new(module: M, rho: F) -> Self {
        Self { module, rho }
    }

    pub fn activate<T>(&self, args: T) -> F::Output
    where
        F: Activate<T>,
    {
        self.rho.activate(args)
    }
}

impl<X, Y, Z, M, F> Predict<X> for Perceptron<M, F>
where
    F: for<'a> Activate<&'a Y, Output = Z>,
    M: Module + Predict<X, Output = Y>,
{
    type Output = Z;

    fn predict(&self, args: &X) -> Result<Self::Output, PredictError> {
        let res = self.module.predict(args)?;
        Ok(self.rho.activate(&res))
    }
}
