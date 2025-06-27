/*
    appellation: impl_init <module>
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
    #[cfg(feature = "rand")]
    pub fn init_rand<G, Dst, Sh>(shape: Sh, distr: G) -> Self
    where
        D: ndarray::RemoveAxis,
        S: ndarray::DataOwned,
        Sh: ndarray::ShapeBuilder<Dim = D>,
        Dst: Clone + rand_distr::Distribution<A>,
        G: Fn(&Sh) -> Dst,
    {
        use crate::init::Initialize;
        let dist = distr(&shape);
        Self::rand(shape, dist)
    }
}
