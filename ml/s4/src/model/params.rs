/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array1, Array2, ArrayView1};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::{vstack, Scalar};
use num::complex::ComplexFloat;

pub struct S4Store<T = f64>
where
    T: ComplexFloat,
{
    pub a: Array2<T>, // Lambda
    pub b: Array2<T>,
    pub c: Array2<T>,
    pub d: Array2<T>,
}

impl<T> S4Store<T>
where
    T: ComplexFloat,
{
    pub fn new(a: Array2<T>, b: Array2<T>, c: Array2<T>, d: Array2<T>) -> Self {
        Self { a, b, c, d }
    }

    pub fn zeros(features: usize) -> Self {
        let a = Array2::<T>::zeros((features, features));
        let b = Array2::<T>::zeros((features, features));
        let c = Array2::<T>::zeros((features, features));
        let d = Array2::<T>::zeros((features, features));
        Self { a, b, c, d }
    }

    pub fn a(&self) -> &Array2<T> {
        &self.a
    }

    pub fn a_mut(&mut self) -> &mut Array2<T> {
        &mut self.a
    }

    pub fn b(&self) -> &Array2<T> {
        &self.b
    }

    pub fn b_mut(&mut self) -> &mut Array2<T> {
        &mut self.b
    }

    pub fn c(&self) -> &Array2<T> {
        &self.c
    }

    pub fn c_mut(&mut self) -> &mut Array2<T> {
        &mut self.c
    }

    pub fn d(&self) -> &Array2<T> {
        &self.d
    }

    pub fn d_mut(&mut self) -> &mut Array2<T> {
        &mut self.d
    }
}

impl<T> S4Store<T>
where
    T: ComplexFloat + Scalar + 'static,
{
    pub fn scan(&self, u: &Array2<T>, x0: &Array1<T>) -> Result<Array2<T>, LinalgError> {
        let step = |xs: &mut Array1<T>, us: ArrayView1<T>| {
            let x1 = self.a.dot(xs) + self.b.t().dot(&us);
            let y1 = self.c.dot(&x1.t());
            Some(y1)
        };
        vstack(
            u.outer_iter()
                .scan(x0.clone(), step)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }
}

// impl<T> ops::Index<SSMParams> for S4Store<T> where T: Float {
//     type Output = Array2<T>;

//     fn index(&self, index: SSMParams) -> &Self::Output {
//         use SSMParams::*;
//         match index {
//             A => &self.a,
//             B => &self.b,
//             C => &self.c,
//             D => &self.d,
//         }
//     }
// }
