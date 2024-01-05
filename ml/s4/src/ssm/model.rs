/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::SSMConfig;
use crate::neural::Forward;
use crate::ops::Discrete;
use crate::params::{SSMParams::*, SSMStore};
use crate::prelude::{discretize, k_convolve};
use ndarray::prelude::{Array1, Array2, Axis, NdFloat};
use ndarray_conv::{Conv2DFftExt, PaddingMode, PaddingSize};
use ndarray_linalg::{Lapack, Scalar};
use num::Float;
use rustfft::FftNum;

pub struct SSM<T = f64>
where
    T: Float,
{
    cache: Array1<T>,
    config: SSMConfig,
    kernel: Array1<T>,
    params: SSMStore<T>,
    ssm: Discrete<T>,
}

impl<T> SSM<T>
where
    T: Float,
{
    pub fn create(config: SSMConfig) -> Self
    where
        T: Default,
    {
        let features = config.features();

        let cache = Array1::<T>::zeros(features);
        let kernel = Array1::<T>::zeros(features);
        let params = SSMStore::from_features(features);
        Self {
            cache,
            config,
            kernel,
            params,
            ssm: Discrete::from_features(features),
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

    pub fn params(&self) -> &SSMStore<T> {
        &self.params
    }

    pub fn params_mut(&mut self) -> &mut SSMStore<T> {
        &mut self.params
    }
}

impl<T> SSM<T>
where
    T: Lapack + NdFloat + Scalar,
{
    pub fn setup(mut self) -> Self {
        self.kernel = self.gen_filter();

        self.ssm = self.discretize(self.config().step_size()).expect("");
        self
    }
}

impl<T> SSM<T>
where
    T: NdFloat + Lapack + Scalar,
{
    pub fn scan(
        &self,
        u: &Array2<T>,
        x0: &Array1<T>,
    ) -> Result<Array2<T>, ndarray_linalg::error::LinalgError> {
        self.params.scan(u, x0)
    }

    pub fn conv(&self, u: &Array2<T>) -> anyhow::Result<Array2<T>>
    where
        T: FftNum,
    {
        let mode = PaddingMode::<2, T>::Const(T::zero());
        let size = PaddingSize::Full;
        if let Some(res) = u.conv_2d_fft(&self.kernel.clone().insert_axis(Axis(1)), size, mode) {
            Ok(res)
        } else {
            Err(anyhow::anyhow!("convolution failed"))
        }
    }

    pub fn discretize(&self, step: T) -> anyhow::Result<Discrete<T>> {
        let discrete = discretize(&self.params[A], &self.params[B], &self.params[C], step)?;
        Ok(discrete.into())
    }

    pub fn gen_filter(&self) -> Array1<T> {
        k_convolve(
            &self.params[A],
            &self.params[B],
            &self.params[C],
            self.config().samples(),
        )
    }
}

impl<T> Forward<Array2<T>> for SSM<T>
where
    T: FftNum + Lapack + NdFloat + Scalar,
{
    type Output = anyhow::Result<Array2<T>>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        let res = if !self.config().decode() {
            self.conv(args)?
        } else {
            self.scan(args, &self.cache)?
        };
        let pred = res + args * &self.params[D];
        Ok(pred)
    }
}
