/*
    Appellation: dplr <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Diagonal Plus Low Rank (DPLR)
//!
//!
use super::utils::*;
use crate::prelude::S4Float;
use ndarray::prelude::{Array1, Array2};
use ndarray::{LinalgScalar, ScalarOperand};
use num::complex::ComplexFloat;
use num::{Complex, Num};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct DPLR<T = f64>
where
    T: Clone + Num,
{
    pub lambda: Array2<Complex<T>>,
    pub p: Array1<T>,
    pub b: Array1<T>,
    pub v: Array2<T>,
}

impl<T> DPLR<T>
where
    T: S4Float,
    Complex<<T as ComplexFloat>::Real>: LinalgScalar + ScalarOperand,
{
    pub fn new(features: usize) -> Self {
        make_dplr_hippo(features)
    }
}
