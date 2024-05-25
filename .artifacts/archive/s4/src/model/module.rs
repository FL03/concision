/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::S4Config;
use crate::cmp::kernel::kernel;
use crate::Conjugate;
use concision::prelude::randcomplex;
use neural::prelude::{Predict, PredictError};
// use crate::prelude::SSMParams::*;
use crate::hippo::dplr::DPLR;
use crate::prelude::{casual_conv1d, discretize_dplr, DPLRParams, SSM};
use ndarray::prelude::{Array, Array1, Axis};
use ndarray::ScalarOperand;
use ndarray_linalg::{flatten, Lapack, Scalar};
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use num::complex::{Complex, ComplexFloat};
use num::traits::{Float, FloatConst, Num, NumOps};
use rustfft::FftNum;

pub struct S4Layer<T = f64> {
    cache: Array1<Complex<T>>,
    config: S4Config,
    kernel: Array1<T>,
    ssm: SSM<Complex<T>>,
}

impl<T> S4Layer<T> {
    pub fn new(config: S4Config) -> Self
    where
        T: Clone + Default + Num,
    {
        let n = config.features();
        let ssm = SSM::from_features(n);
        Self {
            cache: Array::ones((n,)),
            config,
            kernel: Array::ones((n,)),
            ssm,
        }
    }

    pub fn cache(&self) -> &Array1<Complex<T>> {
        &self.cache
    }

    pub fn config(&self) -> &S4Config {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut S4Config {
        &mut self.config
    }
}

impl<T> S4Layer<T>
where
    T: FftNum + Scalar<Complex = Complex<T>, Real = T> + ScalarOperand,
    <T as Scalar>::Complex: Conjugate + Lapack + ScalarOperand,
    <T as Scalar>::Real:
        Conjugate + Float + FloatConst + NumOps<<T as Scalar>::Complex, <T as Scalar>::Complex>,
    StandardNormal: Distribution<T>,
{
    pub fn init(mut self) -> anyhow::Result<Self> {
        let dplr = DPLR::new(self.config.features());
        let c = randcomplex(self.config().features());
        let d = Array::from_elem((1, 1), Complex::new(T::one(), T::zero()));
        if self.config.decode() {
            // RNN Mode
            let discrete = discretize_dplr(
                &dplr.lambda,
                &dplr.p,
                &dplr.p,
                &dplr.b,
                &c,
                self.config().logstep(),
                self.config.samples() as i32,
            )?;
            self.ssm = SSM::new(discrete.a, discrete.b, discrete.c, d);
        } else {
            // CNN Mode
            let params = DPLRParams::new(dplr.lambda, dplr.p.clone(), dplr.p, dplr.b, c);
            self.kernel = kernel(&params, self.config().logstep(), self.config().samples());
        }
        Ok(self)
    }
}

impl<T> Predict<Array1<T>> for S4Layer<T>
where
    T: FftNum + Scalar<Complex = Complex<T>, Real = T> + ScalarOperand,
    Complex<T>: ComplexFloat<Real = T> + Scalar + ScalarOperand,
{
    type Output = Array1<T>;

    fn predict(&self, args: &Array1<T>) -> Result<Self::Output, PredictError> {
        let argsc = args.mapv(|i| i.as_c());
        let u = argsc.clone().insert_axis(Axis(1));
        let mut pred = if !self.config().decode() {
            casual_conv1d(args, &self.kernel)?
        } else {
            let ys = self.ssm.scan(&u, &self.cache)?;
            flatten(ys).view().split_complex().re.to_owned()
        };
        let d = flatten(self.ssm.d().clone());

        pred = &pred + args * d.view().split_complex().re.to_owned();
        Ok(pred)
    }
}
