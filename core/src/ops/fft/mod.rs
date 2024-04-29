/*
   Appellation: fft <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Fast Fourier Transform
//!
//!
pub use self::{fft::*, modes::*, plan::*, utils::*};

pub(crate) mod fft;
pub(crate) mod modes;
pub(crate) mod plan;

pub mod algorithms;

pub trait Fft<T> {
    fn fft(&self) -> Vec<T>;
    fn ifft(&self) -> Vec<T>;
}

pub(crate) mod utils {
    use super::FftPlan;
    use crate::prelude::AsComplex;
    use num::complex::{Complex, ComplexFloat};
    use num::traits::{Float, FloatConst, NumAssignOps, NumCast, NumOps};

    pub(crate) fn fft_angle<T>(n: usize) -> T
    where
        T: FloatConst + NumCast + NumOps,
    {
        T::TAU() / T::from(n).unwrap()
    }

    /// Computes the Fast Fourier Transform of a one-dimensional, complex-valued signal.
    pub fn fft<S, T>(input: impl AsRef<[S]>, permute: &FftPlan) -> Vec<Complex<S::Real>>
    where
        S: ComplexFloat<Real = T>,
        T: Float + FloatConst,
        Complex<T>: ComplexFloat<Real = T> + NumOps<S> + NumOps<T>,
    {
        //
        let input = input.as_ref();
        //
        let n = input.len();
        // initialize the result vector
        let mut result = Vec::with_capacity(n);
        // store the input values in the result vector according to the permutation
        for position in permute.clone().into_iter() {
            let arg = input[position];
            result.push(Complex::new(arg.re(), arg.im()));
        }
        let mut segment: usize = 1;
        while segment < n {
            segment <<= 1;
            // compute the angle of the complex number
            let angle = fft_angle::<T>(segment);
            // compute the radius of the complex number (length)
            let radius = Complex::new(angle.cos(), angle.sin());
            // iterate over the signal in segments of length `segment`
            for start in (0..n).step_by(segment) {
                let mut w = Complex::new(T::one(), T::zero());
                for position in start..(start + segment / 2) {
                    let a = result[position];
                    let b = result[position + segment / 2] * w;
                    result[position] = a + b;
                    result[position + segment / 2] = a - b;
                    w = w * radius;
                }
            }
        }
        result
    }

    /// Computes the Fast Fourier Transform of an one-dimensional, real-valued signal.
    /// TODO: Optimize the function to avoid unnecessary computation.
    pub fn rfft<T>(
        input: impl AsRef<[T]>,
        input_permutation: impl AsRef<[usize]>,
    ) -> Vec<Complex<T>>
    where
        T: Float + FloatConst,
        Complex<T>: ComplexFloat<Real = T> + NumAssignOps,
    {
        // create a reference to the input
        let input = input.as_ref();
        // fetch the length of the input
        let n = input.len();
        // compute the size of the result vector
        let size = (n - (n % 2)) / 2 + 1;
        // initialize the output vector
        let mut store = Vec::with_capacity(size);
        // store the input values in the result vector according to the permutation
        for position in input_permutation.as_ref() {
            store.push(input[*position].as_re());
        }
        let mut segment: usize = 1;
        while segment < n {
            segment <<= 1;
            // compute the angle of the complex number
            let angle = fft_angle::<T>(segment);
            // compute the radius of the complex number (length)
            let radius = Complex::new(angle.cos(), angle.sin());
            for start in (0..n).step_by(segment) {
                let mut w = Complex::new(T::one(), T::zero());
                for position in start..(start + segment / 2) {
                    let a = store[position];
                    let b = store[position + segment / 2] * w;
                    store[position] = a + b;
                    store[position + segment / 2] = a - b;
                    w *= radius;
                }
            }
        }
        store
            .iter()
            .cloned()
            .filter(|x| x.im() >= T::zero())
            .collect()
    }
    /// Computes the Inverse Fast Fourier Transform of an one-dimensional, complex-valued signal.
    pub fn ifft<S, T>(input: &[S], input_permutation: &FftPlan) -> Vec<Complex<T>>
    where
        S: ComplexFloat<Real = T>,
        T: Float + FloatConst,
        Complex<T>: ComplexFloat<Real = T> + NumOps<S> + NumOps<T>,
    {
        let n = input.len();
        let mut result = Vec::with_capacity(n);
        for position in input_permutation.clone().into_iter() {
            let arg = input[position];
            result.push(Complex::new(arg.re(), arg.im()));
        }
        let mut length: usize = 1;
        while length < n {
            length <<= 1;
            let angle = fft_angle::<T>(length).neg();
            let radius = Complex::new(T::cos(angle), T::sin(angle)); // w_len
            for start in (0..n).step_by(length) {
                let mut w = Complex::new(T::one(), T::zero());
                for position in start..(start + length / 2) {
                    let a = result[position];
                    let b = result[position + length / 2] * w;
                    result[position] = a + b;
                    result[position + length / 2] = a - b;
                    w = w * radius;
                }
            }
        }
        let scale = T::from(n).unwrap().recip();
        result.iter().map(|x| *x * scale).collect()
    }
    /// Computes the Inverse Fast Fourier Transform of an one-dimensional, real-valued signal.
    /// TODO: Fix the function; currently fails to compute the correct result
    pub fn irfft<T>(input: &[Complex<T>], plan: &FftPlan) -> Vec<T>
    where
        T: Float + FloatConst,
        Complex<T>: ComplexFloat<Real = T> + NumAssignOps,
    {
        let n = input.len();
        let mut result = vec![Complex::new(T::zero(), T::zero()); n];

        for position in plan.clone().into_iter() {
            result.push(input[position]);
        }
        // for res in result.clone() {
        //     if res.im() > T::zero() {
        //         result.push(res.conj());
        //     }
        // }
        // segment length
        let mut segment: usize = 1;
        while segment < n {
            segment <<= 1;
            // compute the angle of the complex number
            let angle = fft_angle::<T>(segment).neg();
            // compute the radius of the complex number (length)
            let radius = Complex::new(T::cos(angle), T::sin(angle));
            for start in (0..n).step_by(segment) {
                let mut w = Complex::new(T::one(), T::zero());
                for position in start..(start + segment / 2) {
                    let a = result[position];
                    let b = result[position + segment / 2] * w;
                    result[position] = a + b;
                    result[position + segment / 2] = a - b;
                    w *= radius;
                }
            }
        }
        let scale = T::from(n).unwrap().recip();
        result.iter().map(|x| x.re() * scale).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::almost_equal;
    use lazy_static::lazy_static;
    use num::complex::{Complex, ComplexFloat};

    pub(crate) fn fft_permutation(length: usize) -> Vec<usize> {
        let mut result = Vec::new();
        result.reserve_exact(length);
        for i in 0..length {
            result.push(i);
        }
        let mut reverse = 0_usize;
        let mut position = 1_usize;
        while position < length {
            let mut bit = length >> 1;
            while bit & reverse != 0 {
                reverse ^= bit;
                bit >>= 1;
            }
            reverse ^= bit;
            // This is equivalent to adding 1 to a reversed number
            if position < reverse {
                // Only swap each element once
                result.swap(position, reverse);
            }
            position += 1;
        }
        result
    }

    const EPSILON: f64 = 1e-6;

    lazy_static! {
        static ref EXPECTED_RFFT: Vec<Complex<f64>> = vec![
            Complex { re: 28.0, im: 0.0 },
            Complex { re: -4.0, im: 0.0 },
            Complex {
                re: -4.0,
                im: 1.6568542494923806
            },
            Complex {
                re: -4.0,
                im: 4.000000000000001
            },
            Complex {
                re: -3.999999999999999,
                im: 9.656854249492381
            }
        ];
    }

    #[test]
    fn test_plan() {
        let samples = 16;

        let plan = FftPlan::new(samples);
        assert_eq!(plan.plan(), fft_permutation(16).as_slice());
    }

    #[test]
    fn test_rfft() {
        let polynomial = (0..8).map(|i| i as f64).collect::<Vec<_>>();
        let plan = FftPlan::new(polynomial.len());
        println!("Function Values: {:?}", &polynomial);
        println!("Plan: {:?}", &plan);
        let fft = rfft(&polynomial, &plan);
        let mut tmp = fft
            .iter()
            .cloned()
            .filter(|i| i.im() > 0.0)
            .map(|i| i.conj())
            .collect::<Vec<_>>();
        tmp.sort_by(|a, b| a.im().partial_cmp(&b.im()).unwrap());
        println!("FFT: {:?}", &tmp);
        let mut res = fft.clone();
        res.sort_by(|a, b| a.re().partial_cmp(&b.re()).unwrap());
        res.sort_by(|a, b| a.im().partial_cmp(&b.im()).unwrap());
        println!("R: {:?}", &res);
        res.extend(tmp);
        assert!(fft.len() == EXPECTED_RFFT.len());
        for (x, y) in fft.iter().zip(EXPECTED_RFFT.iter()) {
            assert!(almost_equal(x.re(), y.re(), EPSILON));
            assert!(almost_equal(x.im(), y.im(), EPSILON));
        }
        // let plan = FftPlan::new(fft.len());
        let ifft = irfft(&res, &plan);
        println!("Inverse: {:?}", &ifft);
        for (x, y) in ifft.iter().zip(polynomial.iter()) {
            assert!(almost_equal(*x, *y, EPSILON));
        }
    }

    #[test]
    fn small_polynomial_returns_self() {
        let polynomial = vec![1.0f64, 1.0, 0.0, 2.5];
        let permutation = FftPlan::new(polynomial.len());
        let fft = fft(&polynomial, &permutation);
        let ifft = ifft(&fft, &permutation)
            .into_iter()
            .map(|i| i.re())
            .collect::<Vec<_>>();
        for (x, y) in ifft.iter().zip(polynomial.iter()) {
            assert!(almost_equal(*x, *y, EPSILON));
        }
    }

    #[test]
    fn square_small_polynomial() {
        let mut polynomial = vec![1.0f64, 1.0, 0.0, 2.0];
        polynomial.append(&mut vec![0.0; 4]);
        let permutation = FftPlan::new(polynomial.len());
        let mut fft = fft(&polynomial, &permutation);
        fft.iter_mut().for_each(|num| *num *= *num);
        let ifft = ifft(&fft, &permutation)
            .into_iter()
            .map(|i| i.re())
            .collect::<Vec<_>>();
        let expected = [1.0, 2.0, 1.0, 4.0, 4.0, 0.0, 4.0, 0.0, 0.0];
        for (x, y) in ifft.iter().zip(expected.iter()) {
            assert!(almost_equal(*x, *y, EPSILON));
        }
    }

    #[test]
    #[ignore]
    fn square_big_polynomial() {
        // This test case takes ~1050ms on my machine in unoptimized mode,
        // but it takes ~70ms in release mode.
        let n = 1 << 17; // ~100_000
        let mut polynomial = vec![1.0f64; n];
        polynomial.append(&mut vec![0.0f64; n]);
        let permutation = FftPlan::new(polynomial.len());
        let mut fft = fft(&polynomial, &permutation);
        fft.iter_mut().for_each(|num| *num *= *num);
        let ifft = irfft(&fft, &permutation)
            .into_iter()
            .map(|i| i.re())
            .collect::<Vec<_>>();
        let expected = (0..((n << 1) - 1)).map(|i| std::cmp::min(i + 1, (n << 1) - 1 - i) as f64);
        for (&x, y) in ifft.iter().zip(expected) {
            assert!(almost_equal(x, y, EPSILON));
        }
    }
}
