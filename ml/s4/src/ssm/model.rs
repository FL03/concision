/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::SSMConfig;
use crate::neural::prelude::{Predict, PredictError};
use crate::prelude::{casual_conv1d, SSM};
use ndarray::prelude::{Array1, Axis};
use ndarray::ScalarOperand;
use ndarray_linalg::{flatten, Lapack, Scalar};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num::complex::{Complex, ComplexFloat};
use num::traits::real::Real;
use rustfft::FftNum;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct SSMLayer<T = f64> {
    cache: Array1<T>,
    config: SSMConfig,
    kernel: Array1<T>,
    ssm: SSM<T>,
}

impl<T> SSMLayer<T> {
    pub fn new(config: SSMConfig) -> Self
    where
        T: Default,
    {
        Self {
            cache: Array1::<T>::default(config.features()),
            config,
            kernel: Array1::default(config.samples()),
            ssm: SSM::from_features(config.features()),
        }
    }
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

    pub fn ssm(&self) -> &SSM<T> {
        &self.ssm
    }

    pub fn ssm_mut(&mut self) -> &mut SSM<T> {
        &mut self.ssm
    }
}

impl<T> SSMLayer<T>
where
    T: Lapack + Real + Scalar + ScalarOperand,
    StandardNormal: Distribution<T>,
{
    pub fn create(config: SSMConfig) -> anyhow::Result<Self>
    where
        T: Default,
    {
        // initialize the state space model parameters
        let mut ssm = SSM::from_features(config.features()).init(config.features());
        // discretize the state space model
        ssm.discretize(config.logstep())?;
        // initialize the kernal with the convolution of the state space model
        let kernel = ssm.k_conv(config.samples());

        let layer = Self {
            cache: Array1::<T>::zeros(config.features()),
            config,
            kernel,
            ssm,
        };
        Ok(layer)
    }
    /// Initialize the layer
    pub fn init(mut self) -> anyhow::Result<Self>
    where
        T: Default,
    {
        // initialize the state space model parameters
        self.ssm = self.ssm.init(self.config.features());
        self.ssm.discretize(self.config.logstep())?;
        // initialize the kernal with the convolution of the state space model
        self.kernel = self.ssm.k_conv(self.config.samples());

        Ok(self)
    }
}

impl<T> SSMLayer<T>
where
    T: Lapack + Scalar + ScalarOperand,
{
    pub fn discretize(&mut self, step: f64) -> anyhow::Result<()> {
        self.ssm.discretize(step)
    }
}

impl<T> Predict<Array1<T>> for SSMLayer<T>
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
