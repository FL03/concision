/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{arrays::*, assertions::*};

use ndarray::linalg::Dot;
use ndarray::prelude::*;
use ndarray::{IntoDimension, ShapeError};
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand_distr::{Distribution, StandardNormal};
use ndarray_rand::RandomExt;
use num::complex::Complex;
use num::traits::{AsPrimitive, Float, Num, NumCast};
use rand::distributions::uniform::{SampleUniform, Uniform};

pub fn pad<T>(a: impl IntoIterator<Item = T>, pad: usize, value: Option<T>) -> Vec<T>
where
    T: Clone + Default,
{
    let pad = vec![value.unwrap_or_default(); pad];
    let mut res = Vec::from_iter(a);
    res.extend(pad);
    res
}
///
pub fn floor_div<T>(numerator: T, denom: T) -> T
where
    T: Copy + Num,
{
    (numerator - (numerator % denom)) / denom
}

pub fn arange<T>(a: T, b: T, h: T) -> Array1<T>
where
    T: AsPrimitive<usize> + Float,
{
    let n: usize = ((b - a) / h).as_();
    let mut res = Array1::<T>::zeros(n);
    res[0] = a;
    for i in 1..n {
        res[i] = res[i - 1] + h;
    }
    res
}

pub fn genspace<T: NumCast>(features: usize) -> Array1<T> {
    Array1::from_iter((0..features).map(|x| T::from(x).unwrap()))
}

pub fn linarr<T, D>(dim: impl IntoDimension<Dim = D>) -> Result<Array<T, D>, ShapeError>
where
    D: Dimension,
    T: Float,
{
    let dim = dim.into_dimension();
    let n = dim.as_array_view().product();
    Array::linspace(T::one(), T::from(n).unwrap(), n).into_shape(dim)
}

pub fn linspace<T, D>(dim: impl IntoDimension<Dim = D>) -> Result<Array<T, D>, ShapeError>
where
    D: Dimension,
    T: NumCast,
{
    let dim = dim.into_dimension();
    let n = dim.as_array_view().product();
    Array::from_iter((0..n).map(|x| T::from(x).unwrap())).into_shape(dim)
}
/// Raise a matrix to a power
pub fn powmat<T>(a: &Array2<T>, n: usize) -> Array2<T>
where
    T: Clone + Num + 'static,
    Array2<T>: Dot<Array2<T>, Output = Array2<T>>,
{
    if !a.is_square() {
        panic!("Matrix must be square");
    }
    let mut res = Array2::<T>::eye(a.nrows());
    for _ in 0..n {
        res = res.dot(a);
    }
    res
}
///
pub fn randcomplex<T, D>(shape: impl IntoDimension<Dim = D>) -> Array<Complex<T>, D>
where
    D: Dimension,
    T: Copy + Num,
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
/// creates a matrix from the given shape filled with numerical elements [0, n) spaced evenly by 1
pub fn rangespace<T, D>(dim: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    T: Num + NumCast,
{
    let dim = dim.into_dimension();
    let iter = (0..dim.size()).map(|i| T::from(i).unwrap());
    Array::from_shape_vec(dim, iter.collect()).unwrap()
}
/// Round the given value to the given number of decimal places.
pub fn round_to<T: Float>(val: T, decimals: usize) -> T {
    let factor = T::from(10).expect("").powi(decimals as i32);
    (val * factor).round() / factor
}

/// Creates a random array from a uniform distribution using a given key
pub fn seeded_uniform<T, D>(
    key: u64,
    start: T,
    stop: T,
    shape: impl IntoDimension<Dim = D>,
) -> Array<T, D>
where
    D: Dimension,
    T: SampleUniform,
{
    Array::random_using(
        shape,
        Uniform::new(start, stop),
        &mut StdRng::seed_from_u64(key),
    )
}
///
pub fn seeded_stdnorm<T, D>(key: u64, shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    StandardNormal: Distribution<T>,
{
    Array::random_using(shape, StandardNormal, &mut StdRng::seed_from_u64(key))
}
///
pub fn randc_normal<T, D>(key: u64, shape: impl IntoDimension<Dim = D>) -> Array<Complex<T>, D>
where
    D: Dimension,
    T: Copy + Num,
    StandardNormal: Distribution<T>,
{
    let dim = shape.into_dimension();
    let re = seeded_stdnorm(key, dim.clone());
    let im = seeded_stdnorm(key, dim.clone());
    let mut res = Array::zeros(dim);
    ndarray::azip!((re in &re, im in &im, res in &mut res) {
        *res = Complex::new(*re, *im);
    });
    res
}
/// Given a shape, generate a random array using the StandardNormal distribution
pub fn stdnorm<T, D>(shape: impl IntoDimension<Dim = D>) -> Array<T, D>
where
    D: Dimension,
    StandardNormal: Distribution<T>,
{
    Array::random(shape, StandardNormal)
}

pub(crate) mod assertions {
    use ndarray::prelude::{Array, Dimension};
    use ndarray::ScalarOperand;
    use num::traits::{FromPrimitive, Signed};
    use std::fmt::Debug;
    ///
    pub fn assert_atol<T, D>(a: &Array<T, D>, b: &Array<T, D>, tol: T)
    where
        D: Dimension,
        T: Debug + FromPrimitive + PartialOrd + ScalarOperand + Signed,
    {
        let err = (b - a).mapv(|i| i.abs()).mean().unwrap();
        assert!(err <= tol, "Error: {:?}", err);
    }
    /// A function helper for testing that some result is ok
    pub fn assert_ok<T, E>(res: Result<T, E>) -> T
    where
        E: Debug,
    {
        assert!(res.is_ok(), "Error: {:?}", res.err());
        res.unwrap()
    }
    ///
    pub fn assert_approx<T>(a: T, b: T, epsilon: T)
    where
        T: Debug + PartialOrd + Signed,
    {
        let err = (b - a).abs();
        assert!(err < epsilon, "Error: {:?}", err)
    }
    ///
    pub fn almost_equal<T>(a: T, b: T, epsilon: T) -> bool
    where
        T: PartialOrd + Signed,
    {
        (b - a).abs() < epsilon
    }
}

pub(crate) mod arrays {
    use ndarray::prelude::{s, Array, Array1, Array2, Axis};
    use ndarray::{concatenate, RemoveAxis};
    use num::traits::{Num, Zero};
    /// Creates an n-dimensional array from an iterator of n dimensional arrays.
    pub fn concat_iter<D, T>(
        axis: usize,
        iter: impl IntoIterator<Item = Array<T, D>>,
    ) -> Array<T, D>
    where
        D: RemoveAxis,
        T: Clone,
    {
        let mut arr = iter.into_iter().collect::<Vec<_>>();
        let mut out = arr.pop().unwrap();
        for i in arr {
            out = concatenate!(Axis(axis), out, i);
        }
        out
    }
    /// Creates a larger array from an iterator of smaller arrays.
    pub fn stack_iter<T>(iter: impl IntoIterator<Item = Array1<T>>) -> Array2<T>
    where
        T: Clone + Num,
    {
        let mut iter = iter.into_iter();
        let first = iter.next().unwrap();
        let shape = [iter.size_hint().0 + 1, first.len()];
        let mut res = Array2::<T>::zeros(shape);
        res.slice_mut(s![0, ..]).assign(&first);
        for (i, s) in iter.enumerate() {
            res.slice_mut(s![i + 1, ..]).assign(&s);
        }
        res
    }
    ///
    pub fn hstack<T>(iter: impl IntoIterator<Item = Array1<T>>) -> Array2<T>
    where
        T: Clone + Num,
    {
        let iter = Vec::from_iter(iter);
        let mut res = Array2::<T>::zeros((iter.first().unwrap().len(), iter.len()));
        for (i, s) in iter.iter().enumerate() {
            res.slice_mut(s![.., i]).assign(s);
        }
        res
    }
    /// Returns the lower triangular portion of a matrix.
    pub fn tril<T>(a: &Array2<T>) -> Array2<T>
    where
        T: Clone + Zero,
    {
        let mut out = a.clone();
        for i in 0..a.shape()[0] {
            for j in i + 1..a.shape()[1] {
                out[[i, j]] = T::zero();
            }
        }
        out
    }
    /// Returns the upper triangular portion of a matrix.
    pub fn triu<T>(a: &Array2<T>) -> Array2<T>
    where
        T: Clone + Zero,
    {
        let mut out = a.clone();
        for i in 0..a.shape()[0] {
            for j in 0..i {
                out[[i, j]] = T::zero();
            }
        }
        out
    }
    ///
    pub fn vstack<T>(iter: impl IntoIterator<Item = Array1<T>>) -> Array2<T>
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
}

pub(crate) mod linalg {}
