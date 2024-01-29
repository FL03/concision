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

    pub fn config(&self) -> &SSMConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut SSMConfig {
        &mut self.config
    }

    pub fn discretize<S>(&self, step: S) -> anyhow::Result<Discrete<T>>
    where
        S: Scalar<Real = S, Complex = T> + ScalarOperand + NumOps<T, T>,
        T: ComplexFloat<Real = S> + Lapack + NumOps<S>,
    {
        let discrete = discretize(&self.params[A], &self.params[B], &self.params[C], step)?;
        Ok(discrete.into())
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

// impl<T> SSM<T>
// where
//     T: ComplexFloat + Lapack + NumOps<<T as Scalar>::Real> + Scalar<Complex = T>,
//     <T as Scalar>::Real: ScalarOperand + NumOps + NumOps<T, T>,
// {
//     pub fn setup(mut self) -> Self {
//         self.kernel = self.gen_filter();

//         self.ssm = self.discretize(self.config().step_size()).expect("");
//         self
//     }

//     pub fn scan(
//         &self,
//         u: &Array2<T>,
//         x0: &Array1<T>,
//     ) -> Result<Array2<T>, ndarray_linalg::error::LinalgError> {
//         self.params.scan(u, x0)
//     }

//     pub fn conv(&self, u: &Array2<T>) -> anyhow::Result<Array2<T>>
//     where
//         T: FftNum,
//     {
//         let mode = PaddingMode::<2, T>::Const(T::zero());
//         let size = PaddingSize::Full;
//         if let Some(res) = u.conv_2d_fft(&self.kernel.clone().insert_axis(Axis(1)), size, mode) {
//             Ok(res)
//         } else {
//             Err(anyhow::anyhow!("convolution failed"))
//         }
//     }

//     pub fn gen_filter(&self) -> Array1<T> {
//         k_conv(
//             &self.params[A],
//             &self.params[B],
//             &self.params[C],
//             self.config().samples(),
//         )
//     }
// }

// impl<T> Forward<Array2<T>> for SSM<T>
// where
//     T: FftNum + Lapack + NdFloat + Scalar,
// {
//     type Output = anyhow::Result<Array2<T>>;

//     fn forward(&self, args: &Array2<T>) -> Self::Output {
//         let res = if !self.config().decode() {
//             self.conv(args)?
//         } else {
//             self.scan(args, &self.cache)?
//         };
//         let pred = res + args * &self.params[D];
//         Ok(pred)
//     }
// }
