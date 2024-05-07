/*
    Appellation: impl_linear <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Config, Linear, LinearParams};
use core::borrow::{Borrow, BorrowMut};
use nd::RemoveAxis;

impl<T> Linear<T> {
    pub fn std(config: Config) -> Self
    where
        T: Clone + Default,
    {
        let params = LinearParams::default(config.is_biased(), config.dim());
        Self { config, params }
    }
}

impl<T, D> Borrow<Config<D>> for Linear<T, D>
where
    D: RemoveAxis,
{
    fn borrow(&self) -> &Config<D> {
        &self.config
    }
}

impl<T, D> Borrow<LinearParams<T, D>> for Linear<T, D>
where
    D: RemoveAxis,
{
    fn borrow(&self) -> &LinearParams<T, D> {
        &self.params
    }
}

impl<T, D> BorrowMut<LinearParams<T, D>> for Linear<T, D>
where
    D: RemoveAxis,
{
    fn borrow_mut(&mut self) -> &mut LinearParams<T, D> {
        &mut self.params
    }
}
