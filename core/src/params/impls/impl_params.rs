/*
    appellation: impl_params <module>
    authors: @FL03
*/
use crate::params::ParamsBase;

use core::iter::Once;
use ndarray::{RawData, Dimension};


impl<A, S, D> IntoIterator for ParamsBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Item = ParamsBase<S, D>;
    type IntoIter = Once<ParamsBase<S, D>>;

    fn into_iter(self) -> Self::IntoIter {
        core::iter::once(self)
    }
}
