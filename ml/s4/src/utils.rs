/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::{AsComplex, Conjugate};
use ndarray::prelude::*;
use ndarray::{IntoDimension, ScalarOperand};
use ndarray_linalg::Scalar;
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, StandardNormal, Uniform};
use ndarray_rand::RandomExt;
use num::complex::{Complex, ComplexFloat};
use num::traits::FloatConst;
use num::{Float, Num, Signed};
use rustfft::{FftNum, FftPlanner};

pub fn stdnorm<T, D>(shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    StandardNormal: Distribution<T>,
{
    Array::random(shape, StandardNormal)
}

pub fn randcomplex<T, D>(shape: impl IntoDimension<Dim = D>) -> Array<Complex<T>, D>
where
    D: Dimension,
    T: ComplexFloat,
    StandardNormal: Distribution<T>,
{
    let dim = shape.into_dimension();
    let re = Array::random(dim.clone(), StandardNormal);
    let im = Array::random(dim.clone(), StandardNormal);
    let mut res = Array::zeros(dim);
    ndarray::azip!((re in &re, im in &im, res in &mut res) {
        *res = Complex::new(*re, *im);
    });
    res
}

pub fn cauchy<T, D>(v: &Array<T, D>, omega: &Array<T, D>, lambda: &Array<T, D>) -> Array<T, D>
where
    D: Dimension,
    T: Clone + Num + ScalarOperand + Signed,
{
    let cdot = |b: T| (v / (lambda * T::one().neg() + b)).sum();
    omega.mapv(cdot)
}

pub fn cauchy_complex<T, D, S>(
    v: &Array<T, D>,
    omega: &Array<T, S>,
    lambda: &Array<T, D>,
) -> Array<T, S>
where
    D: Dimension,
    S: Dimension,
    T: ComplexFloat + ScalarOperand,
{
    let cdot = |b: T| (v / (lambda * T::one().neg() + b)).sum();
    omega.mapv(cdot)
}

// pub fn eig_sym(args: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
//     let sym = args.clone().into_nalgebra().symmetric_eigen();
//     (
//         sym.eigenvalues.into_ndarray1(),
//         sym.eigenvectors.into_ndarray2(),
//     )
// }

// pub fn eig_csym(args: &Array2<Complex<f64>>) -> (Array1<f64>, Array2<Complex<f64>>) {
//     let sym = args.clone().into_nalgebra().symmetric_eigen();
//     let (eig, v) = (sym.eigenvalues, sym.eigenvectors);
//     (eig.into_ndarray1(), v.into_ndarray2())
// }

// pub fn eigh(args: &Array2<f64>) -> (Array1<f64>, Array2<f64>) {
//     let na = args.clone().into_nalgebra();
//     let sym = na.symmetric_eigen();
//     let v = sym.eigenvectors;
//     let eig = sym.eigenvalues.into_ndarray1();
//     let eigval = v.into_ndarray2();
//     (eig, eigval)
// }

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
    move |shape| logstep(a, b, shape)
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
        let x1 = a.dot(xs) + b.t().dot(&us);
        let y1 = c.dot(&x1.t());
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

pub fn stack_arrays<T>(iter: impl IntoIterator<Item = Array1<T>>) -> Array2<T>
where
    T: Clone + Num,
{
    let iter = Vec::from_iter(iter);
    let mut res = Array2::<T>::zeros((iter.len(), iter.first().unwrap().len()));
    for (i, s) in iter.iter().enumerate() {
        res.slice_mut(s![i, ..]).assign(s);
    }
    res
}

pub fn kernel_dplr<T>(
    lambda: &Array2<T>,
    p: &Array2<T>,
    q: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    step: T,
    l: usize,
) -> Array1<Complex<<T as ComplexFloat>::Real>>
where
    T: AsComplex + ComplexFloat + Conjugate + FloatConst + Scalar + ScalarOperand,
    <T as ComplexFloat>::Real: NdFloat + Num + Signed + num::FromPrimitive + num::Zero,
    <T as Scalar>::Complex: ComplexFloat,
{
    let omega_l = {
        let f = | i: usize | {
            T::from(i).unwrap()
                * T::from(Complex::new(T::one(), -T::PI() / T::from(l).unwrap())).unwrap()
        };
        Array::from_iter((0..l).map(f))
    };
    let aterm = (c.conj(), q.conj());
    let bterm = (b, p);

    let two = T::from(2).unwrap();

    let g = ((&omega_l * T::one().neg() + T::one()) / (&omega_l + T::one())) * (two / step);
    let c = (&omega_l + T::one()).mapv(|i| two / i);

    let k00 = cauchy_complex(&(&aterm.0 * bterm.0), &g, lambda);
    let k01 = cauchy_complex(&(&aterm.0 * bterm.1), &g, lambda);
    let k10 = cauchy_complex(&(&aterm.1 * bterm.0), &g, lambda);
    let k11 = cauchy_complex(&(&aterm.1 * bterm.1), &g, lambda);

    let at_roots = &c * (&k00 - k01 * (&k11 + T::one()).mapv(|i| T::one() / i) * &k10);

    let mut fft_planner = FftPlanner::new();
    let fft = fft_planner.plan_fft_inverse(l);
    let mut buffer = at_roots
        .mapv(|i| Complex::new(i.re(), i.im()))
        .into_raw_vec();
    fft.process(buffer.as_mut_slice());
    Array::from_vec(buffer)
}
