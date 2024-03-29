/*
    Appellation: discretize <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::{Conjugate, Power};

use ndarray::{Array, Array1, Array2, Axis, ScalarOperand};
use ndarray_linalg::{Inverse, Lapack, Scalar};
use num::complex::ComplexFloat;
use num::traits::{Float, NumOps};

pub fn discretize<S, T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    step: S,
) -> anyhow::Result<Discrete<T>>
where
    S: Scalar<Real = S, Complex = T> + ScalarOperand + NumOps<T, T>,
    T: ComplexFloat<Real = S> + Lapack + NumOps<S>,
{
    let (n, ..) = a.dim();
    let hs = step / S::from(2).unwrap(); // half step
    let eye = Array2::<T>::eye(n);

    let bl = (&eye - a * hs).inv()?;

    let ab = bl.dot(&(&eye + a * hs));
    let bb = (bl * step).dot(b);

    Ok((ab, bb, c.clone()).into())
}

pub fn discretize_dplr<S, T>(
    lambda: &Array1<S>,
    p: &Array1<S>,
    q: &Array1<S>,
    b: &Array1<S>,
    c: &Array1<S>,
    step: T,
    l: usize,
) -> anyhow::Result<Discrete<S>>
where
    T: Float + Conjugate + Lapack + NumOps<S, S> + Scalar<Real = T, Complex = S> + ScalarOperand,
    S: ComplexFloat<Real = T> + Conjugate + Lapack + NumOps<T>,
{
    let n = lambda.dim();
    // create an identity matrix; (n, n)
    let eye = Array2::<S>::eye(n);
    // compute the step size
    let hs = T::from(2).unwrap() / step;
    // turn the parameters into two-dimensional matricies
    let b2 = b.clone().insert_axis(Axis(1));

    let c2 = c.clone().insert_axis(Axis(0));

    let p2 = p.clone().insert_axis(Axis(1));
    // compute the conjugate transpose of q
    let qct = q.clone().conj().t().to_owned().insert_axis(Axis(0));
    // create a diagonal matrix D from the scaled eigenvalues: Dim(n, n) :: 1 / (step_size - value)
    let d = Array::from_diag(&lambda.mapv(|i| (hs - i).recip()));

    // create a diagonal matrix from the eigenvalues
    let a = Array::from_diag(&lambda) - &p2.dot(&q.clone().insert_axis(Axis(1)).conj().t());
    // compute A0
    let a0 = &eye * hs + &a;
    // compute A1
    let a1 = {
        let tmp = qct.dot(&d.dot(&p2)).mapv(|i| (T::one() + i).recip());
        &d - (&d.dot(&p2) * tmp * &qct.dot(&d))
    };
    // compute a-bar
    let ab = a1.dot(&a0);
    // compute b-bar
    let bb = a1.dot(&b2) * T::from(2).unwrap();
    // compute c-bar
    let cb = c2.dot(&(&eye - &ab.pow(l)).inv()?.conj());
    // return the discretized system
    Ok((ab, bb, cb.conj()).into())
}

pub trait Discretize<T = f64>
where
    T: Float,
{
    type Output;

    fn discretize(&self, step: T) -> Self::Output;
}

#[derive(Clone, Debug)]
pub struct Discrete<T = f64> {
    pub a: Array2<T>,
    pub b: Array2<T>,
    pub c: Array2<T>,
}

impl<T> Discrete<T> {
    pub fn new(a: Array2<T>, b: Array2<T>, c: Array2<T>) -> Self {
        Self { a, b, c }
    }

    pub fn from_features(features: usize) -> Self
    where
        T: Default,
    {
        let a = Array2::<T>::default((features, features));
        let b = Array2::<T>::default((features, 1));
        let c = Array2::<T>::default((1, features));
        Self::new(a, b, c)
    }
}

impl<T> Discrete<T> {
    pub fn discretize<S>(&self, step: S) -> anyhow::Result<Self>
    where
        S: Scalar<Real = S, Complex = T> + ScalarOperand + NumOps<T, T>,
        T: ComplexFloat<Real = S> + Lapack + NumOps<S>,
    {
        discretize(&self.a, &self.b, &self.c, step)
    }
}

impl<T> From<(Array2<T>, Array2<T>, Array2<T>)> for Discrete<T> {
    fn from((a, b, c): (Array2<T>, Array2<T>, Array2<T>)) -> Self {
        Self::new(a, b, c)
    }
}

impl<T> From<Discrete<T>> for (Array2<T>, Array2<T>, Array2<T>) {
    fn from(discrete: Discrete<T>) -> Self {
        (discrete.a, discrete.b, discrete.c)
    }
}

pub enum DiscretizeArgs<T> {
    DPLR {
        lambda: Array1<T>,
        p: Array1<T>,
        q: Array1<T>,
        b: Array1<T>,
        c: Array1<T>,
    },
}

pub struct Discretizer<T = f64> {
    pub step: T,
}

pub struct Discretized<T>(T);
