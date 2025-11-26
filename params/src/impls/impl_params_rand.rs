/*
    appellation: impl_params_init <module>
    authors: @FL03
*/
use crate::params::ParamsBase;

use ndarray::{DataOwned, Dimension, RawData, RemoveAxis, ScalarOperand, ShapeBuilder};
use num_traits::{Float, FromPrimitive};
use rand_distr::Distribution;

impl<A, S, D> ParamsBase<S, D, A>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// returns a new instance of the [`ParamsBase`] with the given shape and values
    /// initialized according to the provided random distribution `distr`.
    pub fn random_with<Dst, Sh>(shape: Sh, distr: Dst) -> Self
    where
        D: RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        Dst: Clone + Distribution<A>,
    {
        Self::init_from_fn(shape, || distr.sample(&mut rand::rng()))
    }
}
