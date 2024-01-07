/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::*;
use ndarray::{IntoDimension, ScalarOperand};
use ndarray_rand::rand_distr::uniform::SampleUniform;
use ndarray_rand::rand_distr::{Distribution, Uniform};
use ndarray_rand::RandomExt;
use num::complex::{Complex, ComplexDistribution,};
use num::traits::Num;
use std::ops::Neg;

pub fn cauchy<T, A, B>(a: &Array<T, A>, b: &Array<T, B>, c: &Array<T, A>) -> Array<T, B>
where
    A: Dimension,
    B: Dimension,
    T: Num + Neg<Output = T> + ScalarOperand,
{
    let cdot = |b: T| (a / (c * T::one().neg() + b)).sum();
    b.mapv(cdot)
}

pub fn logstep<T, D>(a: T, b: T, shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    T: NdFloat + SampleUniform,
{
    Array::random(shape, Uniform::new(a, b)) * (b.ln() - a.ln()) + a.ln()
}


/// Generate a random array of complex numbers with real and imaginary parts in the range [0, 1)
pub fn randc<T, D>(shape: impl IntoDimension<Dim = D>) -> Array<Complex<T>, D>
where
    D: Dimension,
    T: Distribution<T> + Num,
    ComplexDistribution<T, T>: Distribution<Complex<T>>,
{
    let distr = ComplexDistribution::<T, T>::new(T::one(), T::one());
    Array::random(shape, distr)
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
