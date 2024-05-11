/*
    Appellation: impl_linear <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Config, Linear, LinearParams, ParamMode};
use core::borrow::{Borrow, BorrowMut};
use nd::RemoveAxis;

impl<A, K> Linear<A, K>
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

impl<A, K, D> Borrow<Config<K, D>> for Linear<A, K, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    fn borrow(&self) -> &Config<K, D> {
        &self.config
    }
}

impl<A, K, D> Borrow<LinearParams<K, A, D>> for Linear<A, K, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    fn borrow(&self) -> &LinearParams<K, A, D> {
        &self.params
    }
}

impl<A, K, D> BorrowMut<LinearParams<K, A, D>> for Linear<A, K, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    fn borrow_mut(&mut self) -> &mut LinearParams<K, A, D> {
        &mut self.params
    }
}
