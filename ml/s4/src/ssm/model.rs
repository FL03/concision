/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::SSMConfig;
use crate::neural::Forward;
use crate::ops::Discrete;
use crate::params::{SSMParams::*, SSMStore};
use crate::prelude::discretize;
use ndarray::prelude::Array1;
use ndarray::ScalarOperand;
use ndarray_linalg::{Lapack, Scalar};
use num::complex::ComplexFloat;
use num::traits::NumOps;

pub struct SSM<T = f64> {
    cache: Array1<T>,
    config: SSMConfig,
    kernel: Array1<T>,
    params: SSMStore<T>,
    ssm: Discrete<T>,
}

impl<T> SSM<T> {
    pub fn config(&self) -> &SSMConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut SSMConfig {
        &mut self.config
    }

    pub fn kernel(&self) -> &Array1<T> {
        &self.kernel
    }

    pub fn kernel_mut(&mut self) -> &mut Array1<T> {
        &mut self.kernel
    }

    pub fn params(&self) -> &SSMStore<T> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut SSMStore<T> {
        &mut self.params
    }
}

impl<T> SSM<T> {
    pub fn create(config: SSMConfig) -> Self
    where
        T: Clone + Default,
    {
        let features = config.features();

        let cache = Array1::<T>::default(features);
        let kernel = Array1::<T>::default(features);
        let params = SSMStore::from_features(features);
        Self {
            cache,
            config,
            kernel,
            params,
            ssm: Discrete::from_features(features),
        }
    }

    pub fn discretize<S>(&self, step: S) -> anyhow::Result<Discrete<T>>
    where
        S: Scalar<Real = S, Complex = T> + ScalarOperand + NumOps<T, T>,
        T: ComplexFloat<Real = S> + Lapack + NumOps<S>,
    {
        discretize(&self.params[A], &self.params[B], &self.params[C], step)
    }
}

impl<T> SSM<T> {
    pub fn init(mut self) -> Self {
        self
    }
}
