/*
    appellation: impl_params_init <module>
    authors: @FL03
*/
use crate::params::ParamsBase;

use ndarray::{Dimension, RawData};

impl<A, S, D> ParamsBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
}
