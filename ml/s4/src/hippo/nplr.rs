/*
    Appellation: nplr <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Normal Plus Low Rank (NPLR)
//!
//!
use super::utils::*;

use ndarray::prelude::{Array1, Array2};
use ndarray::ScalarOperand;
use num::complex::ComplexFloat;
// use num::traits::{Float, FloatConst};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NPLR<T = f64> {
    pub a: Array2<T>,
    pub p: Array1<T>,
    pub b: Array1<T>,
}

impl<T> NPLR<T>
where
    T: ComplexFloat + ScalarOperand,
{
    pub fn new(features: usize) -> Self {
        make_nplr_hippo(features)
    }
}

impl<T> From<NPLR<T>> for (Array2<T>, Array1<T>, Array1<T>) {
    fn from(nplr: NPLR<T>) -> Self {
        (nplr.a, nplr.p, nplr.b)
    }
}

impl<T> From<(Array2<T>, Array1<T>, Array1<T>)> for NPLR<T> {
    fn from((a, p, b): (Array2<T>, Array1<T>, Array1<T>)) -> Self {
        Self { a, p, b }
    }
}
