/*
    appellation: impl_params_iter <module>
    authors: @FL03
*/
use crate::params::ParamsBase;

use ndarray::{Dimension, RawData};

#[doc(hidden)]
impl<S, D, A> ParamsBase<S, D>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
}
