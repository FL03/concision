/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{s, Array1, Array2, ArrayView1, NdFloat};
use ndarray::IntoDimension;
use num::{Complex, Float};
use rustfft::{FftNum, FftPlanner};

pub fn cauchy<T>(v: &Array1<T>, omega: &Array1<T>, lambda: &Array1<T>) -> Array1<T>
where
    T: NdFloat,
{
    let cdot = |b: T| (v / (lambda * T::one().neg() + b)).sum();
    omega.mapv(cdot)
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
