/*
    Appellation: linear <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Activate, ActivationFn};
use serde::{Deserialize, Serialize};

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct LinearActivation;

impl LinearActivation {
    pub fn method<T>() -> ActivationFn<T> {
        |x| x
    }

    pub fn rho<T>(args: T) -> T {
        args
    }
}

impl<T> Activate<T> for LinearActivation {
    fn activate(&self, x: T) -> T {
        Self::method()(x)
    }
}
