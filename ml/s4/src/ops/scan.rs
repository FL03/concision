/*
    Appellation: scan <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::SSMStore;
use ndarray::prelude::{Array1, Array2, ArrayView1, NdFloat};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::{vstack, Scalar};
use num::complex::ComplexFloat;
use num::Float;

pub fn scan_ssm<T>(
    a: &Array2<T>,
    b: &Array2<T>,
    c: &Array2<T>,
    u: &Array2<T>,
    x0: &Array1<T>,
) -> Result<Array2<T>, LinalgError>
where
    T: ComplexFloat + NdFloat + Scalar,
{
    let step = |xs: &mut Array1<T>, us: ArrayView1<T>| {
        let x1 = a.dot(xs) + b.dot(&us);
        let y1 = c.dot(&x1);
        Some(y1)
    };
    let scan = u
        .outer_iter()
        .scan(x0.clone(), step)
        .collect::<Vec<Array1<T>>>();
    vstack(scan.as_slice())
}

pub fn scan<F, S, T>(f: &mut F, init: S, xs: Vec<T>) -> (S, Vec<S>)
where
    F: FnMut(&mut S, &T) -> S,
    S: Clone,
    T: Clone,
{
    let mut state = init;
    let mut out = Vec::with_capacity(xs.len());
    for x in xs {
        state = f(&mut state, &x);
        out.push(state.clone());
    }
    (state, out)
}

pub struct Scanner<'a, T = f64>
where
    T: Float,
{
    model: &'a mut SSMStore<T>,
}

impl<'a, T> Scanner<'a, T>
where
    T: Float,
{
    pub fn new(model: &'a mut SSMStore<T>) -> Self {
        Self { model }
    }

    pub fn model(&self) -> &SSMStore<T> {
        self.model
    }

    pub fn model_mut(&mut self) -> &mut SSMStore<T> {
        self.model
    }
}
