/*
    Appellation: convolve <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::ops::{pad, PadMode};
use crate::core::prelude::Power;
use crate::prelude::{irfft, rfft};
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2};
use ndarray::{ErrorKind, ScalarOperand, ShapeError};
use num::complex::{Complex, ComplexFloat};
use num::traits::{Num, NumAssignOps, NumCast};
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

pub fn casual_convolution<T>(u: &Array2<T>, k: &Array1<T>) -> Result<Array2<T>, ShapeError>
where
    T: FftNum + NumCast,
    Complex<T>: ComplexFloat<Real = T> + NumAssignOps,
{
    if u.shape()[0] != k.shape()[0] || u.shape()[1] != k.shape()[0] {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }

    let (m, n) = u.dim();

    let elems = m * (n + k.shape()[0]);
    let shape = [m, n + k.shape()[0]];

    let ud = {
        let padded = pad(&u, &[[0, k.shape()[0]]], PadMode::Constant(T::zero()));
        let fft = rfft::<T>(padded);
        Array::from_shape_vec(shape, fft[0..elems].to_vec())?
    };

    let kd = {
        let padded = pad(&k, &[[0, m]], PadMode::Constant(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };
    println!("{:?}", ud.shape());
    let tmp = &ud * kd;
    println!("E");
    let res = irfft(tmp, elems)[0..m].to_vec();
    let out = Array::from_shape_vec(shape, res)?;
    Ok(out)
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
    let m = u.shape()[0];
    let l2 = m * 2;
    let ud = {
        let padded = pad(&u, &[[0, k.shape()[0]]], PadMode::Constant(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };

    let kd = {
        let padded = pad(&k, &[[0, u.shape()[0]]], PadMode::Constant(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };

    let tmp = &ud * kd;
    let res = irfft(tmp, l2);
    let out = Array::from_vec(res[0..m].to_vec());
    Ok(out)
}

// TODO: Modify the function to work on a one-dimensional kernel
pub fn casual_conv2d<T>(u: &Array2<T>, k: &Array2<T>) -> Result<Array2<T>, ShapeError>
where
    T: FftNum + NumCast,
    Complex<T>: ComplexFloat<Real = T> + NumAssignOps,
{
    if u.shape()[0] != k.shape()[0] || u.shape()[1] != k.shape()[1] {
        return Err(ShapeError::from_kind(ErrorKind::IncompatibleShape));
    }
    let pad_mode = PadMode::Constant(T::zero());
    let (m, n) = u.dim();
    let elems = m * (n + k.shape()[0]);

    let shape = [u.shape()[0], u.shape()[1] + k.shape()[0]];

    let ud = {
        let padded = pad(&u, &[[0, k.shape()[0]]], pad_mode);
        let fft = rfft::<T>(padded);
        Array::from_vec(fft)
        // Array::from_shape_vec(shape, fft[0..elems].to_vec())?
    };

    let kd = {
        let padded = pad(&k, &[[0, u.shape()[0]]], pad_mode);
        let fft = rfft::<T>(padded);
        Array::from_vec(fft)
        // Array::from_shape_vec(shape, fft[0..elems].to_vec())?
    };

    let tmp = &ud * kd;
    let res = irfft(tmp, elems * 2);
    let out = Array::from_shape_vec(shape, res[0..elems].to_vec())?;
    println!("{:?}", out);
    Ok(out)
}

pub struct Filter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::assert_approx;

    use lazy_static::lazy_static;
    use ndarray::prelude::*;

    const EPSILON: f64 = 1e-8;
    const _FEATURES: usize = 4;
    const SAMPLES: usize = 8;

    lazy_static! {
        static ref EXP: Array1<f64> =
            array![-7.10542736e-15, 0.0, 1.0, 4.0, 1.0e1, 2.0e1, 3.5e1, 5.6e1];
        static ref EXP2: Array1<f64> =
            array![0.0, -7.10542736e-15, 1.0, 4.0, 1.0e1, 2.0e1, 3.5e1, 5.6e1];
        static ref EXP2D: Array2<f64> = array![
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [4.0, 0.0, 0.0, 0.0],
        ];
    }

    #[test]
    fn test_casual_conv1d() {
        let u = Array::range(0.0, SAMPLES as f64, 1.0);
        let k = Array::range(0.0, SAMPLES as f64, 1.0);

        let res = casual_conv1d(&u, &k).unwrap();
        for (i, j) in res.into_iter().zip(EXP2.clone().into_iter()) {
            assert_approx(i, j, EPSILON);
        }
    }
    #[ignore = "Casual Convolution on 2D arrays is not yet supported"]
    #[test]
    fn test_casual_conv2d() {
        let samples = 3;
        let u = Array::range(0.0, samples as f64, 1.0).insert_axis(Axis(1));
        let k = Array::range(0.0, samples as f64, 1.0).insert_axis(Axis(1));

        let res = casual_conv2d(&u, &k).unwrap();
        for (i, j) in res.into_iter().zip(EXP2D.clone().into_iter()) {
            assert_approx(i, j, EPSILON);
        }
    }

    #[ignore = "Currently, the function is only able to work with 1d arrays"]
    #[test]
    fn test_casual_convolution() {
        let u = Array::range(0.0, (SAMPLES * SAMPLES) as f64, 1.0)
            .into_shape((SAMPLES, SAMPLES))
            .unwrap();
        let k = Array::range(0.0, SAMPLES as f64, 1.0);

        let res = casual_convolution(&u, &k).unwrap();
        for (i, j) in res.row(0).into_iter().zip(EXP2.clone().into_iter()) {
            assert_approx(*i, j, EPSILON);
        }
    }
}
