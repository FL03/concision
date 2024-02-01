/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::S4Config;
use crate::neural::prelude::{Predict, PredictError};
// use crate::prelude::SSMParams::*;
use crate::prelude::{casual_conv1d, SSM};
use ndarray::prelude::{Array1, Axis};
use ndarray::ScalarOperand;
use ndarray_linalg::{flatten, Scalar};
use num::complex::{Complex, ComplexFloat};
use rustfft::FftNum;

pub struct S4<T = f64> {
    cache: Array1<T>, // make complex
    config: S4Config,
    kernel: Array1<T>,
    ssm: SSM<T>,
}

impl<T> S4<T> {
    pub fn new(config: S4Config) -> Self
    where
        T: Default,
    {
        let n = config.features();
        let cache = Array1::<T>::default((n,));
        let kernel = Array1::<T>::default((n,));
        let store = SSM::from_features(n);
        Self {
            cache,
            config,
            kernel,
            ssm: store,
        }
    }

    pub fn cache(&self) -> &Array1<T> {
        &self.cache
    }

    pub fn config(&self) -> &S4Config {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut S4Config {
        &mut self.config
    }
}

impl<T> S4<T>
where
    T: Scalar,
{
    pub fn setup(self) -> Self {
        self
    }
}

impl<T> Predict<Array1<T>> for S4<T>
where
    T: FftNum + Scalar<Complex = Complex<T>, Real = T> + ScalarOperand,
    Complex<T>: ComplexFloat<Real = T>,
{
    type Output = Array1<T>;

    fn predict(&self, args: &Array1<T>) -> Result<Self::Output, PredictError> {
        let u = args.clone().insert_axis(Axis(1));
        let mut pred = if !self.config().decode() {
            casual_conv1d(args, &self.kernel)?
        } else {
            let ys = self.ssm.scan(&u, &self.cache)?;
            flatten(ys)
        };
        pred = &pred + args * flatten(self.ssm.d().clone());
        Ok(pred)
    }
}
