use crate::params::ParamsBase;

use ndarray::{Data, DataOwned, Dimension, ScalarOperand, ShapeBuilder};
use num_traits::{Float, FromPrimitive};

impl<A, S, D> ParamsBase<S, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A>,
{
    #[cfg(feature = "rand")]
    pub fn init_rand<G, Dst, Sh>(shape: Sh, distr: G) -> Self
    where
        D: ndarray::RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        Dst: Clone + rand_distr::Distribution<A>,
        G: Fn(&Sh) -> Dst,
    {
        use crate::init::Initialize;
        let dist = distr(&shape);
        Self::rand(shape, dist)
    }
}
