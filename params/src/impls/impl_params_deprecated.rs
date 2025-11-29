/*
    appellation: impl_params_iter <module>
    authors: @FL03
*/
use crate::params_base::ParamsBase;

use ndarray::{Dimension, RawData};

#[doc(hidden)]
impl<S, D, A> ParamsBase<S, D, A>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
}
