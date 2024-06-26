/*
    Appellation: dropout <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::Forward;
use ndarray::prelude::{Array, Dimension};
use ndarray::ScalarOperand;
use ndarray_rand::rand_distr::Bernoulli;
use ndarray_rand::RandomExt;
use num::Num;
use serde::{Deserialize, Serialize};

pub fn dropout<T, D>(array: &Array<T, D>, p: f64) -> Array<T, D>
where
    D: Dimension,
    T: Num + ScalarOperand,
{
    // Create a Bernoulli distribution for dropout
    let distribution = Bernoulli::new(p).unwrap();

    // Create a mask of the same shape as the input array
    let mask: Array<bool, D> = Array::random(array.dim(), distribution);
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

    pub fn dropout<T, D>(&self, array: &Array<T, D>) -> Array<T, D>
    where
        D: Dimension,
        T: Num + ScalarOperand,
    {
        dropout(array, self.p)
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(None, 0.5)
    }
}

impl<T, D> Forward<Array<T, D>> for Dropout
where
    D: Dimension,
    T: Num + ScalarOperand,
{
    type Output = Array<T, D>;

    fn forward(&self, input: &Array<T, D>) -> Self::Output {
        dropout(input, self.p)
    }
}
