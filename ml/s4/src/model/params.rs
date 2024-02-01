/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::scan_ssm;
use ndarray::prelude::{Array1, Array2};
use ndarray_linalg::error::LinalgError;
use ndarray_linalg::Scalar;
use num::traits::Num;

pub struct S4Store<T = f64> {
    pub a: Array2<T>, // Lambda
    pub b: Array2<T>,
    pub c: Array2<T>,
    pub d: Array2<T>,
}

impl<T> S4Store<T> {
    pub fn new(a: Array2<T>, b: Array2<T>, c: Array2<T>, d: Array2<T>) -> Self {
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
    T: Clone + Num,
{
    pub fn ones(features: usize) -> Self {
        Self {
            a: Array2::ones((features, features)),
            b: Array2::ones((features, features)),
            c: Array2::ones((features, features)),
            d: Array2::ones((features, features)),
        }
    }

    pub fn zeros(features: usize) -> Self {
        Self {
            a: Array2::zeros((features, features)),
            b: Array2::zeros((features, features)),
            c: Array2::zeros((features, features)),
            d: Array2::zeros((features, features)),
        }
    }
}
impl<T> S4Store<T>
where
    T: Scalar + 'static,
{
    pub fn scan(&self, u: &Array2<T>, x0: &Array1<T>) -> Result<Array2<T>, LinalgError> {
        scan_ssm(&self.a, &self.b, &self.c, u, x0)
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
