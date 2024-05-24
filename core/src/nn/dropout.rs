/*
    Appellation: dropout <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![allow(unused_imports)]
use crate::Forward;
use nd::prelude::*;
use nd::{DataOwned, ScalarOperand};
#[cfg(feature = "rand")]
use ndrand::{rand_distr::Bernoulli, RandomExt};
use num::traits::Num;

#[cfg(feature = "rand")]
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

/// [Dropout] randomly zeroizes elements with a given probability (`p`).
pub trait Dropout {
    type Output;

    fn dropout(&self, p: f64) -> Self::Output;
}

/// The [DropoutLayer] layer is randomly zeroizes inputs with a given probability (`p`).
/// This regularization technique is often used to prevent overfitting.
///
///
/// ### Config
///
/// - (p) Probability of dropping an element
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct DropoutLayer {
    pub(crate) p: f64,
}

/*
 ************* Implementations *************
*/
#[cfg(feature = "rand")]
impl<A, S, D> Dropout for ArrayBase<S, D>
where
    A: Num + ScalarOperand,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn dropout(&self, p: f64) -> Self::Output {
        dropout(self, p)
    }
}

impl DropoutLayer {
    pub fn new(p: f64) -> Self {
        Self { p }
    }

    pub fn scale(&self) -> f64 {
        (1f64 - self.p).recip()
    }
}

impl Default for DropoutLayer {
    fn default() -> Self {
        Self::new(0.5)
    }
}

#[cfg(feature = "rand")]
impl<A, S, D> Forward<ArrayBase<S, D>> for DropoutLayer
where
    A: Num + ScalarOperand,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn forward(&self, input: &ArrayBase<S, D>) -> Self::Output {
        input.dropout(self.p)
    }
}
