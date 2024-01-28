/*
    Appellation: convolve <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
// use crate::core::ops::fft::{rfft, irfft, FftPlan};
use crate::prelude::{irfft, rfft};
use crate::core::prelude::{pad, Power};
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Array1, Array2};
use ndarray::ScalarOperand;
use num::complex::{Complex, ComplexFloat};
use num::traits::{Float, FloatConst, Num, NumAssignOps,};
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
    let ud = {
        let padded = pad(u.clone(), k.len(), Some(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };

    let kd = {
        let padded = pad(k.clone(), l, Some(T::zero()));
        Array::from_vec(rfft::<T>(padded))
    };

    let tmp = &ud * kd;
    let res = irfft(tmp, l);
    Array::from_vec(res[0..l].to_vec())
}

pub struct Filter {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::arange;
    use crate::core::ops::fft::*;

    use lazy_static::lazy_static;
    use ndarray::prelude::*;

    const _FEATURES: usize = 4;
    const SAMPLES: usize = 8;

    lazy_static! {
        static ref EXP: Array1<f64> =
            array![-7.10542736e-15, 0.0, 1.0, 4.0, 1.0e1, 2.0e1, 3.5e1, 5.6e1];
    }

    // #[ignore]
    #[test]
    fn test_casual_convolution() {
        let u = arange(0.0, SAMPLES as f64, 1.0);
        let k = arange(0.0, SAMPLES as f64, 1.0);
        
        let plan = FftPlan::new(SAMPLES);
        // println!("{:?}", rfft(u.clone().into_raw_vec(), plan));
        let res = casual_convolution(&u, &k);
        println!("{:?}", res);
        assert_eq!(res, *EXP);
    }
}
