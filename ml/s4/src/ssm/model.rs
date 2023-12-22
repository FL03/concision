/*
    Appellation: model <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::SSMConfig;
use crate::params::SSMParams::*;
use crate::prelude::{scanner, SSMStore};
use faer::prelude::{FaerMat, IntoFaer, SolverCore};
use faer::IntoNdarray;
use faer_core::zip::ViewMut;
use faer_core::{ComplexField, Conjugate, SimpleEntity};
use ndarray::prelude::{Array1, Array2, NdFloat};
use num::{Float, ToPrimitive};

pub struct SSM<T = f64>
where
    T: Float,
{
    config: SSMConfig,
    kernel: Array2<T>,
    params: SSMStore<T>,
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
        let kernel = Array2::<T>::zeros((features, features));
        let params = SSMStore::from_features(features);
        Self {
            config,
            kernel,
            params,
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
    pub fn scan(&self, u: &Array2<T>, x0: &Array1<T>) -> Array2<T> {
        scanner(&self.params[A], &self.params[B], &self.params[C], u, x0)
    }
}

impl<T> SSM<T>
where
    T: NdFloat + Conjugate + SimpleEntity,
    <T as Conjugate>::Canonical: ComplexField + SimpleEntity + ToPrimitive,
{
    pub fn discretize(&mut self, step: T) -> anyhow::Result<()> {
        let ds = step / T::from(2).unwrap();
        let eye = Array2::<T>::eye(self.config.features());
        let bl = &eye - &self.params[A] * ds;
        let be = {
            let mut tmp = bl.view().into_faer().qr().inverse();
            let arr = &tmp.view_mut().into_ndarray();
            arr.mapv(|i| T::from(i).unwrap())
        };
        let ab = &be.dot(&(&eye + &self.params[A] * ds));
        let bb = (&self.params[B] * ds).dot(&self.params[B].t());

        Ok(())
    }
}
