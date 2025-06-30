/*
    appellation: impl_tensor_init <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use crate::init::Initialize;
use ndarray::{DataOwned, Dimension, RawData, ShapeBuilder};
use rand::RngCore;
use rand_distr::Distribution;

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
}

impl<A, S, D> Initialize<S, D> for TensorBase<S, D>
where
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    fn rand<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
    {
        use rand::{SeedableRng, rngs::SmallRng};
        Self::rand_with(shape, distr, &mut SmallRng::from_rng(&mut rand::rng()))
    }

    fn rand_with<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: RngCore + ?Sized,
        Ds: Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_shape_fn(shape, |_| distr.sample(rng))
    }
}
