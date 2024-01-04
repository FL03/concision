/*
    Appellation: nplr <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Normal Plus Low Rank (NPLR)
//!
//!
use super::utils::*;

use crate::core::prelude::SquareRoot;
use ndarray::prelude::{Array1, Array2};
use ndarray::ScalarOperand;
use num::traits::{Num, NumCast, Signed};
use serde::{Deserialize, Serialize};

fn nplr<T>(features: usize) -> (Array2<T>, Array1<T>, Array1<T>)
where
    T: Num + NumCast + ScalarOperand + Signed + SquareRoot,
{
    let hippo = hippo::<T>(features);

    let base = genspace::<T>(features);
    let p = (&base + (T::one() / T::from(2).unwrap())).mapv(T::sqrt);
    let b = (&base * T::from(2).unwrap() + T::one()).mapv(T::sqrt);
    (hippo, p, b)
}

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
pub struct NPLR<T = f64> {
    pub a: Array2<T>,
    pub p: Array1<T>,
    pub b: Array1<T>,
}

impl<T> NPLR<T>
where
    T: NumCast + ScalarOperand + Signed + SquareRoot,
{
    pub fn new(features: usize) -> Self {
        nplr(features).into()
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
