/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{config::*, layer::*, module::*};

pub(crate) mod config;
pub(crate) mod layer;
pub(crate) mod module;

use crate::params::LinearParams;
use concision::models::Module;
use concision::Forward;
use ndarray::linalg::Dot;
use ndarray::{Array, Array1, Array2, Dimension};

pub struct Linear<T = f64> {
    config: LinearConfig,
    params: LinearParams<T>,
}

impl<T> Module for Linear<T> {
    type Config = LinearConfig;
    type Params = LinearParams<T>;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn params(&self) -> &Self::Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Self::Params {
        &mut self.params
    }
}

impl<A, B, T> Forward<A> for Linear<T>
where
    A: Dot<Array2<T>, Output = B>,
    B: core::ops::Add<Array1<T>, Output = B>,
    T: Clone,
{
    type Output = B;

    fn forward(&self, input: &A) -> Self::Output {
        let wt = self.params().weights().t().to_owned();
        input.dot(&wt)
    }
}
