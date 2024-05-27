/*
   Appellation: perceptron <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::prelude::{Activate, Module, Predict, PredictError};
use nd::{ArrayBase, Data, Dimension};

/// Perceptrons are the fundamental building block of multi-layer perceptrons (MLPs).
/// They are used to model a particular layer within a neural network. Generally speaking,
/// Perceptrons consist of a linear set of parameters and an activation function.
pub struct Perceptron<M, F>
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

impl<A, S, D, M, F> Predict<ArrayBase<S, D>> for Perceptron<M, F>
where
    D: Dimension,
    S: Data<Elem = A>,
    F: Activate<M::Output>,
    M: Module + Predict<ArrayBase<S, D>>,
{
    type Output = F::Output;

    fn predict(&self, args: &ArrayBase<S, D>) -> Result<Self::Output, PredictError> {
        let res = self.module.predict(args)?;
        Ok(self.rho.activate(res))
    }
}

// impl<X, M, F> Predict<X> for Perceptron<M, F>
// where
//     F: Activate<M::Output>,
//     M: Module + Predict<X>,
// {
//     type Output = F::Output;

//     fn predict(&self, args: &X) -> Result<Self::Output, PredictError> {
//         let res = self.module.predict(args)?;
//         Ok(self.rho.activate(res))
//     }
// }
