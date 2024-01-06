/*
    Appellation: discretize <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::{Conjugate, Power};

use ndarray::{Array, Array1, Array2, Axis, ScalarOperand};
use ndarray_linalg::{Inverse, Lapack, Scalar};
use num::complex::ComplexFloat;
use num::Float;

pub fn discretize<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    step: T,
) -> anyhow::Result<Discrete<T>>
where
    T: Lapack + Scalar + ScalarOperand,
{
    let ss = step / T::from(2).unwrap(); // half step
    let eye = Array2::<T>::eye(a.shape()[0]);

    let be = (&eye - a * ss).inv().expect("Could not invert matrix");

    let ab = be.dot(&(&eye + a * ss));
    let bb = (be * ss).dot(b);

    Ok((ab, bb, c.clone()).into())
}


pub fn discretize_dplr<T>(
    lambda: &Array1<T>,
    p: &Array1<T>,
    q: &Array1<T>,
    b: &Array1<T>,
    c: &Array1<T>,
    step: T,
    l: usize,
) -> anyhow::Result<Discrete<T>>
where
    T: ComplexFloat + Conjugate + Lapack + Scalar + ScalarOperand,
{
    let n = lambda.dim();
    // create an identity matrix; (n, n)
    let eye = Array2::<T>::eye(n);
    // compute the step size
    let ss = T::from(2).unwrap() * step.recip();
    // turn the parameters into two-dimensional matricies
    let b2 = b.clone().insert_axis(Axis(1));

    let c2 = c.clone().insert_axis(Axis(1));

    let p2 = p.clone().insert_axis(Axis(1));

    let q2 = q.clone().insert_axis(Axis(1));
    // transpose the c matrix
    let ct = c2.t();
    // compute the conjugate transpose of q
    let qct = q2.conj().t().to_owned();
    // create a diagonal matrix D from the scaled eigenvalues: Dim(n, n) :: 1 / (step_size - value)
    let d = Array::from_diag(&lambda.mapv(|i| (ss - i).recip()));

    // create a diagonal matrix from the eigenvalues
    let a = Array::from_diag(&lambda) - &p2.dot(&q2.conj().t());
    // compute A0
    let a0 = &eye * ss + &a;
    // compute A1
    let a1 = {
        let tmp = qct.dot(&d.dot(&p2)).mapv(|i| (T::one() + i).recip());
        &d - &d.dot(&p2) * &tmp * &qct.dot(&d)
    };
    // compute a-bar
    let ab = a0.dot(&a1);
    // compute b-bar
    let bb = a1.dot(&b2) * T::from(2).unwrap();
    // compute c-bar
    let cb = ct.dot(&(&eye - ab.clone().pow(l)).inv()?.conj()).conj();
    // return the discretized system
    Ok((ab, bb, cb).into())
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
        T: Float,
    {
        let a = Array2::<T>::zeros((features, features));
        let b = Array2::<T>::zeros((features, 1));
        let c = Array2::<T>::zeros((1, features));
        Self::new(a, b, c)
    }
}

impl<T> Discrete<T>
where
    T: Lapack + Scalar + ScalarOperand,
{
    pub fn discretize(&self, args: &Self, step: T) -> anyhow::Result<Self> {
        discretize(&args.a, &args.b, &args.c, step)
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
