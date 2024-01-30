/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::S4Config;
use crate::neural::prelude::Forward;
use crate::prelude::SSMParams::*;
use crate::prelude::{casual_conv2d, SSMStore};
use ndarray::prelude::{Array1, Array2,};
use ndarray_linalg::Scalar;
use num::complex::{Complex, ComplexFloat};
use num::Float;
use rustfft::FftNum;

pub struct S4<T = f64> {
    cache: Array1<T>, // make complex
    config: S4Config,
    kernal: Option<Array2<T>>,
    store: SSMStore<T>,
}

impl<T> S4<T> {
    pub fn new(config: S4Config) -> Self
    where
        T: Default,
    {
        let n = config.features();
        let cache = Array1::<T>::default((n,));
        let kernal = None;
        let store = SSMStore::from_features(n);
        Self {
            cache,
            config,
            kernal,
            store,
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

impl<T> Forward<Array2<T>> for S4<T>
where
    T: FftNum + Scalar<Complex = Complex<T>, Real = T>,
    Complex<T>: ComplexFloat<Real = T>,
{
    type Output = Result<Array2<T>, anyhow::Error>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        // let u = args.insert_axis(Axis(1));
        let mut pred = if !self.config().decode() {
            casual_conv2d(args, self.kernal.as_ref().unwrap())?
        } else {
            self.store.scan(args, &self.cache)?
        };
        pred = &pred + args * &self.store[D];
        Ok(pred)
    }
}
