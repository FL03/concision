/*
    Appellation: kernel <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array1;

pub struct DPLRParams<T = f64> {
    pub lambda: Array1<T>,
    pub p: Array1<T>,
    pub q: Array1<T>,
    pub b: Array1<T>,
    pub c: Array1<T>,
}

impl<T> DPLRParams<T> {
    pub fn new(lambda: Array1<T>, p: Array1<T>, q: Array1<T>, b: Array1<T>, c: Array1<T>) -> Self {
        Self { lambda, p, q, b, c }
    }
}

// impl<T> DPLRParams<T>
// where
//     T: ComplexFloat,
//     <T as ComplexFloat>::Real: NumOps + NumOps<Complex<<T as ComplexFloat>::Real>, Complex<<T as ComplexFloat>::Real>>,
//     Complex<<T as ComplexFloat>::Real>: NumOps + NumOps<<T as ComplexFloat>::Real, Complex<<T as ComplexFloat>::Real>>
// {
//     pub fn kernel(&self, step: T, l: usize) -> Array1<<T as ComplexFloat>::Real> {
//         let lt = T::from(l).unwrap();
//         let omega_l = {
//             let f = | i: usize | -> Complex<<T as ComplexFloat>::Real> {
//                 Complex::<<T as ComplexFloat>::Real>::i().neg() * <T as ComplexFloat>::Real::from(i).unwrap() * <T as ComplexFloat>::Real::PI() / lt
//             };
//             Array::from_iter((0..l).map(f))
//         };
//     }
// }

impl<T> From<(Array1<T>, Array1<T>, Array1<T>, Array1<T>, Array1<T>)> for DPLRParams<T> {
    fn from((lambda, p, q, b, c): (Array1<T>, Array1<T>, Array1<T>, Array1<T>, Array1<T>)) -> Self {
        Self::new(lambda, p, q, b, c)
    }
}
