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

pub trait Fft {
    fn fft(&self) -> Vec<f64>;
    fn ifft(&self) -> Vec<f64>;
}

pub(crate) mod utils {
    use super::FftPlan;
    use crate::prelude::AsComplex;
    use num::complex::{Complex, ComplexFloat};
    use num::traits::{Float, FloatConst, NumOps};

    pub(crate) fn fft_angle<T>(n: usize) -> T
    where
        T: Float + FloatConst,
    {
        T::TAU() / T::from(n).unwrap()
    }

    /// Computes the Fast Fourier Transform of a general signal.
    pub fn fft<T>(input: impl AsRef<[T]>, input_permutation: impl AsRef<[usize]>) -> Vec<Complex<T>>
    where
        T: Float + FloatConst,
        Complex<T>: ComplexFloat<Real = T>,
    {
        let input = input.as_ref();

        let n = input.len();

        let mut result = Vec::new();
        result.reserve_exact(n);
        for position in input_permutation.as_ref() {
            result.push(input[*position].as_re());
        }
        let mut segment_length = 1_usize;
        while segment_length < n {
            segment_length <<= 1;
            let angle = fft_angle::<T>(segment_length);
            let w_len = Complex::new(angle.cos(), angle.sin());
            for segment_start in (0..n).step_by(segment_length) {
                let mut w = Complex::new(T::one(), T::zero());
                for position in segment_start..(segment_start + segment_length / 2) {
                    let a = result[position];
                    let b = result[position + segment_length / 2] * w;
                    result[position] = a + b;
                    result[position + segment_length / 2] = a - b;
                    w = w * w_len;
                }
            }
        }
        result
    }

    /// Computes the Fast Fourier Transform of a real-valued signal.
    /// TODO: Fix the function; real-valued fft only computes the positive frequency terms
    pub fn rfft<T>(input: impl AsRef<[T]>, input_permutation: impl AsRef<[usize]>) -> Vec<Complex<T>>
    where
        T: Float + FloatConst,
        Complex<T>: ComplexFloat<Real = T>,
    {
        let input = input.as_ref();

        let n = input.len();

        let mut result = Vec::new();
        result.reserve_exact(n);
        for position in input_permutation.as_ref() {
            result.push(input[*position].as_re());
        }
        let mut segment_length = 1_usize;
        while segment_length < n {
            segment_length <<= 1;
            let angle = fft_angle::<T>(segment_length);
            let w_len = Complex::new(angle.cos(), angle.sin());
            for segment_start in (0..n).step_by(segment_length) {
                let mut w = Complex::new(T::one(), T::zero());
                for position in segment_start..(segment_start + segment_length / 2) {
                    let a = result[position];
                    let b = result[position + segment_length / 2] * w;
                    result[position] = a + b;
                    result[position + segment_length / 2] = a - b;
                    w = w * w_len;
                }
            }
        }
        result
    }
    /// Computes the Inverse Fast Fourier Transform of a signal.
    pub fn ifft<T>(input: &[Complex<T>], input_permutation: &FftPlan) -> Vec<Complex<T>>
    where
        T: Float + FloatConst,
        Complex<T>: ComplexFloat<Real = T>,
    {
        let n = input.len();
        let mut result = Vec::new();
        result.reserve_exact(n);
        for position in input_permutation.clone().into_iter() {
            result.push(input[position]);
        }
        let mut segment_length = 1_usize;
        while segment_length < n {
            segment_length <<= 1;
            let angle = fft_angle::<T>(segment_length);
            let w_len = Complex::new(T::cos(angle), T::sin(angle));
            for segment_start in (0..n).step_by(segment_length) {
                let mut w = Complex::new(T::one(), T::zero());
                for position in segment_start..(segment_start + segment_length / 2) {
                    let a = result[position];
                    let b = result[position + segment_length / 2] * w;
                    result[position] = a + b;
                    result[position + segment_length / 2] = a - b;
                    w = w * w_len;
                }
            }
        }
        let scale = T::from(n).unwrap().recip();
        result.iter().map(|x| x * scale).collect()
    }

    pub fn irfft<T>(input: &[Complex<T>], input_permutation: &FftPlan) -> Vec<T>
    where
        T: Float + FloatConst + NumOps<Complex<T>, Complex<T>>,
        Complex<T>: ComplexFloat<Real = T>,
    {
        let n = input.len();
        let mut result = Vec::new();
        result.reserve_exact(n);
        for position in input_permutation.clone().into_iter() {
            result.push(input[position]);
        }
        let mut segment_length = 1_usize;
        while segment_length < n {
            segment_length <<= 1;
            let angle = fft_angle::<T>(segment_length);
            let w_len = Complex::new(T::cos(angle), T::sin(angle));
            for segment_start in (0..n).step_by(segment_length) {
                let mut w = Complex::new(T::one(), T::zero());
                for position in segment_start..(segment_start + segment_length / 2) {
                    let a = result[position];
                    let b = result[position + segment_length / 2] * w;
                    result[position] = a + b;
                    result[position + segment_length / 2] = a - b;
                    w = w * w_len;
                }
            }
        }
        let scale = T::from(n).unwrap().recip();
        result.iter().map(|x| x.re() * scale).collect()
    }
    /// Computes the Inverse Fast Fourier Transform of a real-valued signal.
    pub fn ifftr<S, T>(input: &[S], input_permutation: &FftPlan) -> Vec<T>
    where
        T: Float + FloatConst,
        S: ComplexFloat<Real = T> + NumOps<Complex<T>>,
    {
        let n = input.len();
        let mut result = Vec::new();
        result.reserve_exact(n);
        for position in input_permutation.clone().into_iter() {
            result.push(input[position]);
        }
        let mut segment_length = 1_usize;
        while segment_length < n {
            segment_length <<= 1;
            let angle = fft_angle::<T>(segment_length);
            let w_len = Complex::new(T::cos(angle), T::sin(angle));
            for segment_start in (0..n).step_by(segment_length) {
                let mut w = S::one();
                for position in segment_start..(segment_start + segment_length / 2) {
                    let a = result[position];
                    let b = result[position + segment_length / 2] * w;
                    result[position] = a + b;
                    result[position + segment_length / 2] = a - b;
                    w = w * w_len;
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
    use num::complex::ComplexFloat;

    pub(crate) fn fast_fourier_transform_input_permutation(length: usize) -> Vec<usize> {
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

    #[test]
    fn test_plan() {
        let samples = 16;

        let plan = FftPlan::new(samples);
        assert_eq!(
            plan.plan(),
            fast_fourier_transform_input_permutation(16).as_slice()
        );
    }

    #[test]
    fn small_polynomial_returns_self() {
        let polynomial = vec![1.0f64, 1.0, 0.0, 2.5];
        let permutation = FftPlan::new(polynomial.len());
        let fft = fft(&polynomial, &permutation);
        let ifft = irfft(&fft, &permutation)
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
        let ifft = irfft(&fft, &permutation)
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
