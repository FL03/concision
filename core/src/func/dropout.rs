/*
    Appellation: dropout <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]
use crate::Forward;
use nd::prelude::*;
use nd::{DataOwned, RemoveAxis, ScalarOperand};
use ndrand::rand_distr::Bernoulli;
use ndrand::RandomExt;
use num::traits::Num;

pub fn dropout<A, S, D>(array: &ArrayBase<S, D>, p: f64) -> Array<A, D>
where
    A: Num + ScalarOperand,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    // Create a Bernoulli distribution for dropout
    let distribution = Bernoulli::new(p).unwrap();

    // Create a mask of the same shape as the input array
    let mask: Array<bool, D> = Array::random(array.dim(), distribution);
    let mask = mask.mapv(|x| if x { A::zero() } else { A::one() });

    // Element-wise multiplication to apply dropout
    array * mask
}

pub fn dropout_axis<A, S, D>(array: &ArrayBase<S, D>, _axis: Axis, p: f64) -> Array<A, D>
where
    A: Num + ScalarOperand,
    D: RemoveAxis,
    S: DataOwned<Elem = A>,
{
    // Create a Bernoulli distribution for dropout
    let distribution = Bernoulli::new(p).unwrap();

    // Create a mask of the same shape as the input array
    let _mask: Array<bool, D> = Array::random(array.dim(), distribution);

    unimplemented!()
}

/// The [Dropout] layer is randomly zeroizes inputs with a given probability (`p`).
/// This regularization technique is often used to prevent overfitting.
///
///
/// ### Config
///
/// - (p) Probability of dropping an element
pub struct Dropout {
    p: f64,
}

impl Dropout {
    pub fn new(p: f64) -> Self {
        Self { p }
    }

    pub fn dropout<A, S, D>(&self, array: &ArrayBase<S, D>) -> Array<A, D>
    where
        A: Num + ScalarOperand,
        D: Dimension,
        S: DataOwned<Elem = A>,
    {
        dropout(array, self.p)
    }

    pub fn scale(&self) -> f64 {
        (1f64 - self.p).recip()
    }
}

impl Default for Dropout {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl<A, S, D> Forward<ArrayBase<S, D>> for Dropout
where
    A: Num + ScalarOperand,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn forward(&self, input: &ArrayBase<S, D>) -> Self::Output {
        dropout(input, self.p)
    }
}
