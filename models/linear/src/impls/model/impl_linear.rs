/*
    Appellation: impl_linear <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Config, Linear, ParamMode, ParamsBase};
use core::borrow::{Borrow, BorrowMut};
use nd::{DataOwned, Ix2, RawData, RemoveAxis};

impl<A, K, S> Linear<A, K, Ix2, S>
where
    K: ParamMode,
    S: RawData<Elem = A>,
{
    pub fn from_features(inputs: usize, outputs: usize) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
    {
        let config = Config::std(inputs, outputs);
        let params = ParamsBase::new(config.dim());
        Self { config, params }
    }
}

impl<A, S, D, K> Borrow<Config<K, D>> for Linear<A, K, D, S>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    fn borrow(&self) -> &Config<K, D> {
        &self.config
    }
}

impl<A, S, D, K> Borrow<ParamsBase<S, D, K>> for Linear<A, K, D, S>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    fn borrow(&self) -> &ParamsBase<S, D, K> {
        &self.params
    }
}

impl<A, S, D, K> BorrowMut<ParamsBase<S, D, K>> for Linear<A, K, D, S>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    fn borrow_mut(&mut self) -> &mut ParamsBase<S, D, K> {
        &mut self.params
    }
}
