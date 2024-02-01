/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::SSMConfig;
use crate::neural::prelude::{Predict, PredictError};
use crate::params::{SSMParams::*, SSM};
use crate::prelude::{casual_conv1d, k_conv, Discrete};
use ndarray::prelude::{Array1, Axis};
use ndarray::ScalarOperand;
use ndarray_linalg::{flatten, Lapack, Scalar};
use num::complex::{Complex, ComplexFloat};
use rustfft::FftNum;

pub struct SSMLayer<T = f64> {
    cache: Array1<T>,
    config: SSMConfig,
    kernel: Array1<T>,
    params: SSM<T>,
    ssm: Discrete<T>,
}

impl<T> SSMLayer<T> {
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

    pub fn params(&self) -> &SSM<T> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut SSM<T> {
        &mut self.params
    }
}

impl<T> SSMLayer<T>
where
    T: Lapack + Scalar + ScalarOperand,
{
    pub fn create(config: SSMConfig) -> anyhow::Result<Self>
    where
        T: Default,
    {
        // initialize the state space model parameters
        let params = SSM::from_features(config.features());
        // discretize the state space model
        let ssm = params.discretize(config.step_size())?;
        // initialize the kernal with the convolution of the state space model
        let kernel = k_conv(&ssm.a, &ssm.b, &ssm.c, config.samples());

        let layer = Self {
            cache: Array1::<T>::zeros(config.features()),
            config,
            kernel,
            params,
            ssm,
        };
        Ok(layer)
    }
}

impl<T> SSMLayer<T>
where
    T: Lapack + Scalar + ScalarOperand,
{
    pub fn discretize(&self, step: f64) -> anyhow::Result<Discrete<T>> {
        Discrete::discretize(&self.params[A], &self.params[B], &self.params[C], step)
    }
}

impl<T> Predict<Array1<T>> for SSMLayer<T>
where
    T: FftNum + Scalar<Complex = Complex<T>, Real = T>,
    Complex<T>: ComplexFloat<Real = T>,
{
    type Output = Array1<T>;

    fn predict(&self, args: &Array1<T>) -> Result<Self::Output, PredictError> {
        let u = args.clone().insert_axis(Axis(1));
        let mut pred = if !self.config().decode() {
            casual_conv1d(args, &self.kernel)?
        } else {
            let ys = self.params.scan(&u, &self.cache)?;
            flatten(ys)
        };
        pred = &pred + args * flatten(self.params.d().clone());
        Ok(pred)
    }
}
