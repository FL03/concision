/*
    Appellation: convolve <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::{pad, Power};
use crate::prelude::{irfft, rfft};
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2, Axis};
use ndarray::{ErrorKind, ScalarOperand, ShapeError};
use num::complex::{Complex, ComplexFloat};
use num::traits::{Float, Num, NumAssignOps, NumCast};
use rustfft::FftNum;

/// Generates a large convolution kernal
pub fn k_conv<T>(a: &Array2<T>, b: &Array2<T>, c: &Array2<T>, l: usize) -> Array1<T>
where
    T: Num + ScalarOperand,
    Array2<T>: Dot<Array2<T>, Output = Array2<T>>,
{
    let f = |i: usize| c.dot(&a.pow(i).dot(b));

    let mut store = Vec::new();
    for i in 0..l {
        store.extend(f(i));
    }
    Array::from_vec(store)
}

///
pub fn casual_conv1d<T>(u: &Array1<T>, k: &Array1<T>) -> Result<Array1<T>, ShapeError>
where
    T: FftNum + NumCast,
    Complex<T>: ComplexFloat<Real = T> + NumAssignOps,
{
    if u.shape()[0] != k.shape()[0] {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }
    let l = u.shape()[0];
    let l2 = l * 2;
    let ud = {
        let padded = pad(u.clone(), k.len(), Some(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };

    let kd = {
        let padded = pad(k.clone(), l, Some(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };

    let tmp = &ud * kd;
    let res = irfft(tmp, l2);
    let out = Array::from_vec(res[0..l].to_vec());
    Ok(out)
}

pub fn casual_conv2d<T>(u: &Array2<T>, k: &Array2<T>) -> Result<Array2<T>, ShapeError>
where
    T: FftNum + NumCast,
    Complex<T>: ComplexFloat<Real = T> + NumAssignOps,
{
    if u.shape()[0] != k.shape()[0] {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }
    let l = u.shape()[0];
    let l2 = l * 2;
    let ud = {
        let padded = pad(u.clone(), k.shape()[0], Some(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };

    let kd = {
        let padded = pad(k.clone(), l, Some(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };

    let tmp = &ud * kd;
    let res = irfft(tmp, l2);
    let out = Array::from_vec(res[0..l].to_vec()).insert_axis(Axis(1));
    Ok(out)
}

pub struct Filter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::assert_approx;

    use lazy_static::lazy_static;
    use ndarray::prelude::*;

    const _FEATURES: usize = 4;
    const SAMPLES: usize = 8;

    lazy_static! {
        static ref EXP: Array1<f64> =
            array![-7.10542736e-15, 0.0, 1.0, 4.0, 1.0e1, 2.0e1, 3.5e1, 5.6e1];
        static ref EXP2: Array1<f64> =
            array![0.0, -7.10542736e-15, 1.0, 4.0, 1.0e1, 2.0e1, 3.5e1, 5.6e1];
    }

    #[test]
    fn test_casual_convolution() {
        let u = Array::range(0.0, SAMPLES as f64, 1.0);
        let k = Array::range(0.0, SAMPLES as f64, 1.0);

        let res = casual_conv1d(&u, &k).unwrap();
        for (i, j) in res.into_iter().zip(EXP2.clone().into_iter()) {
            assert_approx(i, j, 1e-8);
        }
    }
}
