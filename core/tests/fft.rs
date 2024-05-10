/*
    Appellation: fft <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;

use approx::assert_abs_diff_eq;
use concision::ops::fft::*;
use lazy_static::lazy_static;
use num::complex::{Complex, ComplexFloat};

const EPSILON: f64 = 1e-6;

fn fft_permutation(length: usize) -> Vec<usize> {
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
#[ignore = "Needs to be fixed"]
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
        assert_abs_diff_eq!(x.re(), y.re());
        assert_abs_diff_eq!(x.im(), y.im());
    }
    // let plan = FftPlan::new(fft.len());
    let ifft = dbg!(irfft(&res, &plan));
    for (x, y) in ifft.iter().zip(polynomial.iter()) {
        assert_abs_diff_eq!(*x, *y, epsilon = EPSILON);
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
        assert_abs_diff_eq!(*x, *y, epsilon = EPSILON);
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
        assert_abs_diff_eq!(*x, *y, epsilon = EPSILON);
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
        assert_abs_diff_eq!(x, y);
    }
}
