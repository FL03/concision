/*
    Appellation: impl_linear <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Config, Linear, LinearParams, ParamMode};
use core::borrow::{Borrow, BorrowMut};
use nd::RemoveAxis;

impl<K, A> Linear<K, A>
where
    K: ParamMode,
{
    pub fn from_features(inputs: usize, outputs: usize) -> Self
    where
        A: Clone + Default,
    {
        let config = Config::std(inputs, outputs);
        let params = LinearParams::default(config.dim());
        Self { config, params }
    }
}

impl<K, A, D> Borrow<Config<D, K>> for Linear<K, A, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    fn borrow(&self) -> &Config<D, K> {
        &self.config
    }
}

impl<K, A, D> Borrow<LinearParams<K, A, D>> for Linear<K, A, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    fn borrow(&self) -> &LinearParams<K, A, D> {
        &self.params
    }
}

impl<K, A, D> BorrowMut<LinearParams<K, A, D>> for Linear<K, A, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    fn borrow_mut(&mut self) -> &mut LinearParams<K, A, D> {
        &mut self.params
    }
}
