/*
    appellation: impl_params_init <module>
    authors: @FL03
*/
use crate::params::ParamsBase;

use ndarray::{Dimension, RawData, ScalarOperand};
use num_traits::{Float, FromPrimitive};

impl<A, S, D> ParamsBase<S, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: Dimension,
    S: RawData<Elem = A>,
{
}
