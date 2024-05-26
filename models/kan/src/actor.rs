/*
    Appellation: actor <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::config::ActorConfig;

pub(crate) mod config;

use concision::prelude::{Activate, Predict, PredictError};
use core::ops::Mul;
use nd::prelude::*;
use splines::interpolate::{Interpolate, Interpolator};
use splines::Spline;

/// An `actor` describe the learnable activation functions
/// employed throughout KAN networks.
///
/// The learned functions are one-dimensional, continuous
pub struct Actor<T, V, B> {
    pub(crate) bias: B,
    pub(crate) omega: Array1<V>,
    pub(crate) spline: Spline<T, V>,
}

impl<T, V, B> Actor<T, V, B>
where
    B: Activate<V>,
{
    pub fn new(bias: B, omega: Array1<V>, spline: Spline<T, V>) -> Self {
        Self {
            bias,
            omega,
            spline,
        }
    }

    pub fn bias(&self) -> &B {
        &self.bias
    }

    pub fn omega(&self) -> &Array1<V> {
        &self.omega
    }

    pub fn spline(&self) -> &Spline<T, V> {
        &self.spline
    }
}

impl<T, V, B> Predict<Array1<T>> for Actor<T, V, B>
where
    B: Activate<Array1<V>>,
    T: Interpolator,
    V: Interpolate<T>,
    Array1<V>: Clone + Mul<B::Output>,
{
    type Output = <Array1<V> as Mul<B::Output>>::Output;

    fn predict(&self, x: &Array1<T>) -> Result<Self::Output, PredictError> {
        let y = x.mapv(|xi| self.spline.sample(xi).unwrap());
        Ok(self.omega.clone() * self.bias.activate(y))
    }
}
