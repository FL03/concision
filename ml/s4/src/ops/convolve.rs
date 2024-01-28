/*
    Appellation: convolve <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
// use crate::core::ops::fft::{rfft, irfft, FftPlan};
use crate::core::prelude::{floor_div, pad, Power};
use crate::prelude::{irfft, rfft};
use ndarray::linalg::Dot;
use ndarray::prelude::{array, s, Array, Array1, Array2, Axis};
use ndarray::ScalarOperand;
use num::complex::{Complex, ComplexFloat};
use num::traits::{Float, FloatConst, Num, NumAssignOps};
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

// pub fn casual_convolution<T>(u: &Array1<T>, k: &Array1<T>) -> Array1<T>
// where
//     T: Float + FloatConst,
//     Complex<T>: ComplexFloat<Real = T> + NumAssignOps,
// {
//     assert!(u.shape()[0] == k.shape()[0]);
//     let l = u.shape()[0];
//     let plan = FftPlan::new(l);
//     let ud = rfft::<T>(u.clone().into_raw_vec(), &plan);
//     let kd = rfft::<T>(k.clone().into_raw_vec(), &plan);

//     let ud = Array::from_vec(ud);
//     let kd = Array::from_vec(kd);

//     let tmp = ud * kd;
//     let res = irfft(tmp.into_raw_vec().as_slice(), &plan);
//     Array::from_vec(res)
// }

pub fn casual_convolution<T>(u: &Array1<T>, k: &Array1<T>) -> Array1<T>
where
    T: Default + FftNum + Float + FloatConst,
    Complex<T>: ComplexFloat<Real = T> + NumAssignOps,
{
    assert!(u.shape()[0] == k.shape()[0]);
    let l = u.shape()[0];
    println!("{:?}", l);
    let l2 = l * 2;
    let inv_size = floor_div(l, 2) + 1;
    let ud = {
        let padded = pad(u.clone(), k.len(), Some(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };

    let kd = {
        let padded = pad(k.clone(), l, Some(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };

    let mut tmp = &ud * kd;
    // let a = array![Complex::new(T::zero(), T::zero())];
    // tmp.append(Axis(0), a.view()).expect("");
    let res = irfft(tmp, l2);
    // let res = irfft(tmp.slice(s![0..l]).to_vec(), l2);
    Array::from_vec(res[0..l].to_vec())
}

pub struct Filter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::ops::fft::*;
    use crate::core::prelude::{arange, assert_approx};

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

    // #[ignore]
    #[test]
    fn test_casual_convolution() {
        let u = arange(0.0, SAMPLES as f64, 1.0);
        let k = arange(0.0, SAMPLES as f64, 1.0);

        let _plan = FftPlan::new(SAMPLES);
        let res = casual_convolution(&u, &k);
        println!("{:?}", res);
        for (i, j) in res.into_iter().zip(EXP2.clone().into_iter()) {
            assert_approx(i, j, 1e-8);
        }
    }
}
