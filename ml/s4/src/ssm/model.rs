/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::SSMConfig;
use crate::neural::Forward;
use crate::params::{SSMParams::*, SSMStore};
use crate::prelude::{discretize, k_convolve, scanner};
use ndarray::prelude::{Array1, Array2, NdFloat};
use ndarray_conv::{Conv2DFftExt, PaddingMode, PaddingSize};
use num::Float;
use rustfft::FftNum;

#[derive(Clone, Debug)]
pub struct Discrete<T = f64> {
    pub a: Array2<T>,
    pub b: Array2<T>,
    pub c: Array2<T>,
}

impl<T> Discrete<T>
where
    T: Float,
{
    pub fn new(a: Array2<T>, b: Array2<T>, c: Array2<T>) -> Self {
        Self { a, b, c }
    }

    pub fn from_features(features: usize) -> Self
    where
        T: Default,
    {
        let a = Array2::<T>::eye(features);
        let b = Array2::<T>::zeros((features, 1));
        let c = Array2::<T>::zeros((features, features));
        Self { a, b, c }
    }
}

impl<T> From<(Array2<T>, Array2<T>, Array2<T>)> for Discrete<T>
where
    T: Float,
{
    fn from((a, b, c): (Array2<T>, Array2<T>, Array2<T>)) -> Self {
        Self { a, b, c }
    }
}

impl<T> From<Discrete<T>> for (Array2<T>, Array2<T>, Array2<T>)
where
    T: Float,
{
    fn from(discrete: Discrete<T>) -> Self {
        (discrete.a, discrete.b, discrete.c)
    }
}

pub struct SSM<T = f64>
where
    T: Float,
{
    cache: Array1<T>,
    config: SSMConfig,
    kernel: Array2<T>,
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
        let kernel = Array2::<T>::zeros((features, features));
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

    pub fn kernel(&self) -> &Array2<T> {
        &self.kernel
    }

    pub fn kernel_mut(&mut self) -> &mut Array2<T> {
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
    T: NdFloat,
{
    pub fn setup(mut self) -> Self {
        self.kernel = self.gen_filter();

        self.ssm = self.discretize(self.config().step_size()).expect("");
        self
    }
}

impl<T> SSM<T>
where
    T: NdFloat,
{
    pub fn scan(&self, u: &Array2<T>, x0: &Array1<T>) -> Array2<T> {
        scanner(&self.params[A], &self.params[B], &self.params[C], u, x0)
    }

    pub fn conv(&self, u: &Array2<T>) -> anyhow::Result<Array2<T>>
    where
        T: FftNum,
    {
        let mode = PaddingMode::<2, T>::Const(T::zero());
        let size = PaddingSize::Full;
        if let Some(res) = u.conv_2d_fft(&self.kernel, size, mode) {
            Ok(res)
        } else {
            Err(anyhow::anyhow!("convolution failed"))
        }
    }

    pub fn discretize(&self, step: T) -> anyhow::Result<Discrete<T>> {
        let discrete = discretize(&self.params[A], &self.params[B], &self.params[C], step)?;
        Ok(discrete.into())
    }

    pub fn gen_filter(&self) -> Array2<T> {
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
    T: FftNum + NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Array2<T> {
        let res = if !self.config().decode() {
            self.conv(args).expect("convolution failed")
        } else {
            self.scan(args, &self.cache)
        };
        res + args * &self.params[D]
    }
}
