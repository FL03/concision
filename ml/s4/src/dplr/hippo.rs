/*
    Appellation: hippo <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::utils::*;
use ndarray::prelude::Array2;
use ndarray::ScalarOperand;
use num::complex::ComplexFloat;
use num::Float;

pub enum HiPPOs<T = f64> {
    HiPPO(Array2<T>),
    NPLR {
        a: Array2<T>,
        p: Array2<T>,
        b: Array2<T>,
    },
    DPLR {
        lambda: Array2<T>,
        p: Array2<T>,
        q: Array2<T>,
        b: Array2<T>,
        c: Array2<T>,
    },
}

pub struct HiPPO<T = f64>(Array2<T>);

impl<T> HiPPO<T>
where
    T: Float,
{
    pub fn new(hippo: Array2<T>) -> Self {
        Self(hippo)
    }

    pub fn hippo(&self) -> &Array2<T> {
        &self.0
    }

    pub fn hippo_mut(&mut self) -> &mut Array2<T> {
        &mut self.0
    }
}

impl<T> HiPPO<T>
where
    T: ComplexFloat + ScalarOperand,
{
    pub fn square(features: usize) -> Self {
        Self(make_hippo(features))
    }

    pub fn nplr(features: usize) -> Self {
        let (hippo, p, b) = make_nplr_hippo(features);
        Self(hippo)
    }
}
