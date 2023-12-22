/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::{AsComplex, Conjugate};
use nalgebra::ComplexField;
use ndarray::prelude::*;
use ndarray::IntoDimension;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use nshare::{ToNalgebra, ToNdarray1, ToNdarray2};
use num::traits::FloatConst;
use num::{Complex, Float, Zero};
use rustfft::{FftNum, FftPlanner};

pub fn cauchy<T>(v: &Array1<T>, omega: &Array1<T>, lambda: &Array1<T>) -> Array1<T>
where
    T: NdFloat,
{
    let cdot = |b: T| (v / (lambda * T::one().neg() + b)).sum();
    omega.mapv(cdot)
}

pub fn eig_sym(args: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let sym = args.clone().into_nalgebra().symmetric_eigen();
    (
        sym.eigenvalues.into_ndarray1(),
        sym.eigenvectors.into_ndarray2(),
    )
}

pub fn eig_csym(args: &Array2<Complex<f64>>) -> (Array1<f64>, Array2<Complex<f64>>) {
    let sym = args.clone().into_nalgebra().symmetric_eigen();
    let (eig, v) = (sym.eigenvalues, sym.eigenvectors);
    (eig.into_ndarray1(), v.into_ndarray2())
}

pub fn eigh(args: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
    let na = args.clone().into_nalgebra();
    let sym = na.symmetric_eigen();
    let v = sym.eigenvectors;
    let eig = sym.eigenvalues.into_ndarray1();
    let eigval = v.into_ndarray2();
    (eig, eigval)
}

pub fn powmat<T>(a: &Array2<T>, n: usize) -> Array2<T>
where
    T: Float + 'static,
{
    if !a.is_square() {
        panic!("Matrix must be square");
    }
    let mut res = a.clone();
    for _ in 1..n {
        res = res.dot(a);
    }
    res
}

pub fn casual_colvolution<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: FftNum,
{
    let mut planner = FftPlanner::<T>::new();
    let fft = planner.plan_fft_forward(a.len());

    a.clone()
}

pub fn logstep<T, D>(a: T, b: T, shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    T: NdFloat + SampleUniform,
{
    Array::random(shape, Uniform::new(a, b)) * (b.ln() - a.ln()) + a.ln()
}

pub fn logstep_init<T, D>(a: T, b: T) -> impl Fn(D) -> Array<T, D>
where
    D: Dimension,
    T: NdFloat + SampleUniform,
{
    move |shape| Array::random(shape, Uniform::new(a, b)) * (b.ln() - a.ln()) + a.ln()
}

pub fn scanner<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    u: &Array2<T>,
    x0: &Array1<T>,
) -> Array2<T>
where
    T: NdFloat,
{
    let step = |xs: &mut Array1<T>, us: ArrayView1<T>| {
        let x1 = a.dot(xs) + b.dot(&us);
        let y1 = c.dot(&x1);
        Some(y1)
    };
    let scan = u.outer_iter().scan(x0.clone(), step).collect::<Vec<_>>();
    let shape = [scan.len(), scan[0].len()];
    let mut res = Array2::<T>::zeros(shape.into_dimension());
    for (i, s) in scan.iter().enumerate() {
        res.slice_mut(s![i, ..]).assign(s);
    }
    res
}

pub fn scan_complex<T>(
    a: &Array2<Complex<T>>,
    b: &Array2<T>,
    c: &Array2<Complex<T>>,
    u: &Array2<T>,
    x0: &Array1<Complex<T>>,
) -> Array2<Complex<T>>
where
    T: NdFloat,
{
    let step = |xs: &mut Array1<Complex<T>>, us: ArrayView1<T>| {
        let x1 = a.dot(xs) + b.dot(&us);
        let y1 = c.dot(&x1);
        Some(y1)
    };
    let scan = u.outer_iter().scan(x0.clone(), step).collect::<Vec<_>>();
    let shape = [scan.len(), scan[0].len()];
    let mut res = Array2::zeros(shape.into_dimension());
    for (i, s) in scan.iter().enumerate() {
        res.slice_mut(s![i, ..]).assign(s);
    }
    res
}

pub fn scan_ssm_step<'a, T>(
    a: &'a Array2<T>,
    b: &'a Array2<T>,
    c: &'a Array2<T>,
) -> impl FnMut(&'a mut Array1<T>, ArrayView1<'a, T>) -> Option<Array1<T>>
where
    T: NdFloat,
{
    |xs, us| {
        let x1 = a.dot(xs) + b.dot(&us);
        let y1 = c.dot(&x1);
        Some(y1)
    }
}

// fn kernel_dplr<T>(lambda: T, p: &Array2<T>, q: &Array2<T>, b: &Array2<T>, c: &Array2<T>, step: T, l: usize) -> Array1<Complex<T>>
// where
//     T: Conjugate + FloatConst + FftNum + NdFloat,
// {
//     let omega_l: Array1<Complex<T>> = (0..l)
//         .map(|i| Complex::from_polar(T::one(), -T::PI() * T::from(i).unwrap() / T::from(l).unwrap()))
//         .collect();

//     let aterm = (c.conj(), q.conj());
//     let bterm = (b, p);

//     let g = (T::from(2.0).unwrap() / step) * ((T::one().as_complex() - &omega_l) / (&omega_l + T::one()));
//     let c = T::from(2.0).unwrap() / (T::from(1.0).unwrap() + &omega_l);

//     let k00 = cauchy(&aterm.0 * bterm.0, &g, lambda);
//     let k01 = cauchy(&aterm.0 * bterm.1, &g, lambda);
//     let k10 = cauchy(&aterm.1 * bterm.0, &g, lambda);
//     let k11 = cauchy(&aterm.1 * bterm.1, &g, lambda);
//     let at_roots = &c * (&k00 - k01 * (T::one() / (&k11 + T::one())) * &k10);
//     let mut fft_planner = FftPlanner::new();
//     let fft = fft_planner.plan_fft(l, rustfft::FftDirection::Forward);
//     let mut at_roots_complex = Array1::zeros(l);
//     for (i, val) in at_roots.iter().enumerate() {
//         at_roots_complex[i] = Complex::new(*val, T::zero());
//     }
//     let mut out = Array1::zeros(l);
//     fft.process(&mut out);
//     out
// }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::round_to;
    use approx::assert_relative_eq;
    use ndarray::prelude::array;

    #[test]
    fn test_eig_sym() {
        let a = array![[1.0, 2.0], [2.0, 1.0]];
        let (eig, eigval) = eig_sym(&a);
        let exp_eig = array![3.0, -1.0];
        let exp_eigval = array![[0.70710678, -0.70710678], [0.70710678, 0.70710678]];

        assert_relative_eq!(eig, exp_eig);
        assert_relative_eq!(eigval.mapv(|i| round_to(i, 8)), exp_eigval);
    }
}
