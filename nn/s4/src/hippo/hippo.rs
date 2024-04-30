/*
    Appellation: hippo <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
// use super::dplr::DPLR;
use super::nplr::NPLR;
use crate::core::prelude::genspace;
use ndarray::prelude::Array2;
use ndarray::ScalarOperand;
use ndarray_linalg::Scalar;
use serde::{Deserialize, Serialize};

pub enum HiPPOs<T = f64> {
    HiPPO(HiPPO<T>),
    // DPLR(DPLR<T>),
    NPLR(NPLR<T>),
}

impl<T> HiPPOs<T>
where
    T: Scalar + ScalarOperand,
{
    pub fn new(features: usize) -> Self {
        Self::HiPPO(HiPPO::new(features))
    }

    pub fn nplr(features: usize) -> Self {
        Self::NPLR(NPLR::new(features))
    }
}

// impl<T> HiPPOs<T>
// where
//     T: S4Float,
//     Complex<<T as ComplexFloat>::Real>: LinalgScalar + ScalarOperand,
// {
//     pub fn dplr(features: usize) -> Self {
//         Self::DPLR(DPLR::new(features))
//     }
// }

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct HiPPO<T = f64> {
    features: usize,
    data: Array2<T>,
}

impl<T> HiPPO<T> {
    pub fn features(&self) -> usize {
        self.features
    }
}

impl<T> HiPPO<T>
where
    T: Scalar + ScalarOperand,
{
    pub fn new(features: usize) -> Self {
        Self {
            features,
            data: super::hippo(features),
        }
    }

    pub fn nplr(&self) -> NPLR<T> {
        let base = genspace(self.features());
        let p = (&base + T::one() / T::from(2).unwrap()).mapv(T::sqrt);
        let b = (&base * T::from(2).unwrap() + T::one()).mapv(T::sqrt);
        NPLR {
            a: self.as_ref().clone(),
            p,
            b,
        }
    }
}

// impl<T> HiPPO<T>
// where
//     T: S4Float,
// {
//     pub fn dplr(&self) -> DPLR<T> {

//     }
// }

impl<T> AsRef<Array2<T>> for HiPPO<T> {
    fn as_ref(&self) -> &Array2<T> {
        &self.data
    }
}

impl<T> AsMut<Array2<T>> for HiPPO<T> {
    fn as_mut(&mut self) -> &mut Array2<T> {
        &mut self.data
    }
}

impl<T> From<Array2<T>> for HiPPO<T> {
    fn from(a: Array2<T>) -> Self {
        Self {
            features: a.dim().0,
            data: a,
        }
    }
}

impl<T> From<HiPPO<T>> for Array2<T> {
    fn from(hippo: HiPPO<T>) -> Self {
        hippo.data
    }
}
