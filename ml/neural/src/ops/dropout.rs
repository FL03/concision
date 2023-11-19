/*
    Appellation: dropout <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array, Ix1};
use ndarray_rand::rand_distr::Bernoulli;
use ndarray_rand::RandomExt;
use num::Float;
use serde::{Deserialize, Serialize};

pub fn dropout<T>(array: &Array<T, Ix1>, p: f64) -> Array<T, Ix1>
where
    T: Float,
{
    // Create a Bernoulli distribution for dropout
    let distribution = Bernoulli::new(p).unwrap();

    // Create a mask of the same shape as the input array
    let mask: Array<bool, _> = Array::<bool, Ix1>::random(array.dim(), distribution);
    let mask = mask.mapv(|x| if x { T::zero() } else { T::one() });

    // Element-wise multiplication to apply dropout
    array * mask
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, PartialOrd, Serialize)]
pub struct Dropout {
    axis: Option<usize>,
    p: f64,
}

impl Dropout {
    pub fn new(axis: Option<usize>, p: f64) -> Self {
        Self { axis, p }
    }

    pub fn apply<T>(&self, array: &Array<T, Ix1>) -> Array<T, Ix1>
    where
        T: Float,
    {
        dropout(array, self.p)
    }

    pub fn dropout<T>(&self, array: &Array<T, Ix1>, p: f64) -> Array<T, Ix1>
    where
        T: Float,
    {
        // Create a Bernoulli distribution for dropout
        let distribution = Bernoulli::new(p).unwrap();

        // Create a mask of the same shape as the input array
        let mask: Array<bool, _> = Array::<bool, Ix1>::random(array.dim(), distribution);
        let mask = mask.mapv(|x| if x { T::zero() } else { T::one() });

        // Element-wise multiplication to apply dropout
        array * mask
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(None, 0.5)
    }
}
