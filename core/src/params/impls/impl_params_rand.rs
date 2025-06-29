/*
    appellation: impl_params_init <module>
    authors: @FL03
*/
use crate::params::ParamsBase;

use crate::init::Initialize;
use ndarray::{
    ArrayBase, Axis, DataOwned, Dimension, RawData, RemoveAxis, ScalarOperand, ShapeBuilder,
};
use num_traits::{Float, FromPrimitive};
use rand::rngs::SmallRng;
use rand_distr::Distribution;

impl<A, S, D> ParamsBase<S, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// generates a randomly initialized set of parameters with the given shape using the
    /// output of the given distribution function `G`
    pub fn init_rand<G, Dst, Sh>(shape: Sh, distr: G) -> Self
    where
        D: RemoveAxis,
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        Dst: Clone + Distribution<A>,
        G: Fn(&Sh) -> Dst,
    {
        let dist = distr(&shape);
        Self::rand(shape, dist)
    }
}

impl<A, S, D> Initialize<S, D> for ParamsBase<S, D>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
    {
        use rand::SeedableRng;
        Self::rand_with(shape, distr, &mut SmallRng::from_rng(&mut rand::rng()))
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: rand::RngCore + ?Sized,
        Ds: Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
    {
        let shape = shape.into_shape_with_order();
        let bias_shape = shape.raw_dim().remove_axis(Axis(0));
        let bias = ArrayBase::from_shape_fn(bias_shape, |_| distr.sample(rng));
        let weights = ArrayBase::from_shape_fn(shape, |_| distr.sample(rng));
        Self { bias, weights }
    }
}
