/*
    Appellation: discretize <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::Conjugate;
use crate::prelude::powmat;

use ndarray::{Array2, ScalarOperand};
use ndarray_linalg::{Inverse, Lapack, Scalar};
use num::Float;

pub fn discretize<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    step: T,
) -> anyhow::Result<(Array2<T>, Array2<T>, Array2<T>)>
where
    T: Lapack + Scalar + ScalarOperand,
{
    let ss = step / T::from(2).unwrap(); // half step
    let eye = Array2::<T>::eye(a.shape()[0]);

    let be = (&eye - a * ss).inv().expect("Could not invert matrix");

    let ab = be.dot(&(&eye + a * ss));
    let bb = (b * ss).dot(&b.t());

    Ok((ab, bb, c.clone()))
}

pub fn discretize_dplr<T>(
    lambda: &Array2<T>,
    p: &Array2<T>,
    q: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    step: T,
    l: usize,
) -> anyhow::Result<(Array2<T>, Array2<T>, Array2<T>)>
where
    T: Conjugate + Float + Lapack + Scalar + ScalarOperand,
{
    let (n, _m) = lambda.dim();

    let eye = Array2::<T>::eye(n);
    let ss = T::from(2).unwrap() * step.recip();

    let a = {
        let tmp = Array2::from_diag(&lambda.diag());
        tmp - &p.dot(&q.conj().t())
    };

    let a0 = &eye * ss + &a;

    let d = {
        let tmp = lambda.mapv(|i| (ss - i).recip());
        Array2::from_diag(&tmp.diag())
    };

    let qc = {
        let tmp = q.conj();
        tmp.t().to_owned()
    };
    let p2 = p.clone();

    let a1 = {
        let tmp = qc.dot(&d.dot(&p2)).mapv(|i| (T::one() + i).recip());
        &d - &d.dot(&p2) * &tmp * &qc.dot(&d)
    };

    let ab = a0.dot(&a1);
    let bb = a1.dot(b) * T::from(2).unwrap();
    let cb = {
        let tmp = (&eye - powmat(&ab, l)).inv()?.conj();
        c.dot(&tmp)
    };

    Ok((ab, bb, cb.conj()))
}

pub trait Discretize<T = f64>
where
    T: Float,
{
    type Output;

    fn discretize(&self, step: T) -> Self::Output;
}

pub enum DiscretizeArgs {}

pub struct Discretizer<T = f64> {
    pub step: T,
}

pub struct Discretized<T>(T);
