/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{s, Array1, Array2, ArrayView1, NdFloat};
use ndarray::IntoDimension;
use num::Float;
use rustfft::{FftNum, FftPlanner};

pub fn powmat<T>(a: &Array2<T>, n: usize) -> Array2<T>
where
    T: Float + 'static,
{
    if !a.is_square() {
        panic!("Matrix must be square");
    }
    let mut res = a.clone();
    for _ in 0..n {
        res = res.dot(a);
    }
    res
}

pub fn casual_colvolution<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: FftNum,
{
    let mut planner = FftPlanner::<T>::new();
    let fft = planner.plan_fft_forward(a.len());

    a.clone()
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
        let x1 = a.dot(xs) + b.dot(&us);
        let y1 = c.dot(&x1);
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
