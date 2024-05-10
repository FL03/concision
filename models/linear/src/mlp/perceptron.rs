/*
   Appellation: perceptron <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::prelude::{Module, Predict, PredictError};

pub struct Perceptron<F, M>
where
    M: Module,
{
    module: M,
    rho: F,
}

impl<F, M> Perceptron<F, M>
where
    M: Module,
{
    pub fn new(module: M, rho: F) -> Self {
        Self { module, rho }
    }

    pub fn activate<T>(&self, args: &T) -> T
    where
        F: Fn(&T) -> T,
    {
        (self.rho)(args)
    }
}

impl<X, Y, F, M> Predict<X> for Perceptron<F, M>
where
    F: for<'a> Fn(&'a Y) -> Y,
    M: Predict<X, Output = Y> + Module,
{
    type Output = Y;

    fn predict(&self, args: &X) -> Result<Self::Output, PredictError> {
        let res = self.module.predict(args)?;
        Ok(self.activate(&res))
    }
}
