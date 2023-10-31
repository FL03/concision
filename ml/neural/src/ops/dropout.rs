/*
    Appellation: dropout <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Ix1};
use ndarray_rand::rand_distr::Bernoulli;
use ndarray_rand::RandomExt;

pub fn dropout<T>(array: &Array<T, Ix1>, p: f64) -> Array<T, Ix1>
where
    T: num::Float,
{
    // Create a Bernoulli distribution for dropout
    let distribution = Bernoulli::new(p).unwrap();

    // Create a mask of the same shape as the input array
    let mask: Array<bool, _> = Array::<bool, Ix1>::random(array.dim(), distribution);
    let mask = mask.mapv(|x| if x { T::zero() } else { T::one() });

    // Element-wise multiplication to apply dropout
    array * mask
}
