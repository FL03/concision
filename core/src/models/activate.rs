/*
   Appellation: model <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{Module, Predict, PredictError};

pub struct Activator<F, M> {
    module: M,
    rho: F,
}

impl<F, M> Activator<F, M>
where
    F: for<'a> Fn(&'a M::Output) -> M::Output,
    M: Predict<<M as Module>::Params> + Module,
{
    pub fn new(module: M, rho: F,) -> Self {
        Self { module, rho }
    }

    pub fn activate(&self, args: &M::Output) -> M::Output {
        (self.rho)(args)
    }
}

impl<F, M> Predict<M::Params> for Activator<F, M>
where
    F: for<'a> Fn(&'a M::Output) -> M::Output,
    M: Predict<<M as Module>::Params> + Module,
{
    type Output = M::Output;

    fn predict(&self, args: &M::Params) -> Result<Self::Output, PredictError> {
        let res = self.module.predict(args)?;
        Ok(self.activate(&res))
    }
}
