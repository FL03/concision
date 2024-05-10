/*
    Appellation: impl_linear <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Config, Linear, LinearParams};
use core::borrow::{Borrow, BorrowMut};
use nd::RemoveAxis;

impl<A> Linear<A> {
    pub fn from_features(biased: bool, inputs: usize, outputs: usize) -> Self where A: Default {
        let config = Config::from_features(biased, inputs, outputs);
        let params = LinearParams::default(biased, config.dim());
        Self { config, params }
    }
}

impl<A, D> Borrow<Config<D>> for Linear<A, D>
where
    D: RemoveAxis,
{
    fn borrow(&self) -> &Config<D> {
        &self.config
    }
}

impl<A, D> Borrow<LinearParams<A, D>> for Linear<A, D>
where
    D: RemoveAxis,
{
    fn borrow(&self) -> &LinearParams<A, D> {
        &self.params
    }
}

impl<A, D> BorrowMut<LinearParams<A, D>> for Linear<A, D>
where
    D: RemoveAxis,
{
    fn borrow_mut(&mut self) -> &mut LinearParams<A, D> {
        &mut self.params
    }
}
