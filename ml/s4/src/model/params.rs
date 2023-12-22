/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{Array1, Array2, NdFloat};
use num::{Complex, Float};

pub struct S4Store<T = f64>
where
    T: Float,
{
    pub a: Array2<Complex<T>>, // Lambda
    pub b: Array2<T>,
    pub c: Array2<Complex<T>>,
    pub d: Array2<T>,
}

impl<T> S4Store<T>
where
    T: Float,
{
    pub fn new(a: Array2<Complex<T>>, b: Array2<T>, c: Array2<Complex<T>>, d: Array2<T>) -> Self {
        Self { a, b, c, d }
    }

    pub fn zeros(features: usize) -> Self {
        let a = Array2::<Complex<T>>::zeros((features, features));
        let b = Array2::<T>::zeros((features, features));
        let c = Array2::<Complex<T>>::zeros((features, features));
        let d = Array2::<T>::zeros((features, features));
        Self { a, b, c, d }
    }

    pub fn a(&self) -> &Array2<Complex<T>> {
        &self.a
    }

    pub fn a_mut(&mut self) -> &mut Array2<Complex<T>> {
        &mut self.a
    }

    pub fn b(&self) -> &Array2<T> {
        &self.b
    }

    pub fn b_mut(&mut self) -> &mut Array2<T> {
        &mut self.b
    }

    pub fn c(&self) -> &Array2<Complex<T>> {
        &self.c
    }

    pub fn c_mut(&mut self) -> &mut Array2<Complex<T>> {
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
    T: NdFloat,
{
    pub fn scan(&self, u: &Array2<T>, x0: &Array1<Complex<T>>) -> Array2<Complex<T>> {
        crate::scan_complex(&self.a, &self.b, &self.c, u, x0)
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
