/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::S4Config;
use crate::neural::prelude::Forward;
use crate::prelude::SSMStore;
use ndarray::prelude::{Array1, Array2, NdFloat};
use ndarray_conv::{Conv2DFftExt, PaddingMode, PaddingSize};
use num::Float;
use rustfft::FftNum;

use crate::prelude::SSMParams::*;

pub struct S4<T = f64>
where
    T: Float,
{
    cache: Array1<T>,
    config: S4Config,
    kernal: Option<Array2<T>>,
    store: SSMStore<T>,
}

impl<T> S4<T>
where
    T: Float,
{
    pub fn new(config: S4Config) -> Self
    where
        T: Default,
    {
        let n = config.features();
        let cache = Array1::<T>::zeros((n,));
        let kernal = None;
        let store = SSMStore::from_features(n);
        Self {
            cache,
            config,
            kernal,
            store,
        }
    }
}

impl<T> Forward<Array2<T>> for S4<T>
where
    T: FftNum + NdFloat,
{
    type Output = Array2<T>;

    fn forward(&self, args: &Array2<T>) -> Self::Output {
        if !self.config.decode {
            let mode = PaddingMode::<2, T>::Const(T::zero());
            let size = PaddingSize::Full;
            return args
                .conv_2d_fft(&self.kernal.clone().unwrap(), size, mode)
                .expect("convolution failed");
        }
        let scan = self.store.scan(args, &self.cache);
        scan + args * &self.store[D]
    }
}
