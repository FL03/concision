/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::{s, Array1, Array2, ArrayView1, Axis, NdFloat};
use ndarray::IntoDimension;
use rustfft::{FftNum, FftPlanner};

pub fn casual_colvolution<T>(a: &Array2<T>, b: &Array2<T>) -> Array2<T>
where
    T: FftNum,
{
    let mut planner = FftPlanner::<T>::new();
    let fft = planner.plan_fft_forward(a.len());

    a.clone()
}

pub fn k_convolve<T>(a: &Array2<T>, b: &Array2<T>, c: &Array2<T>, l: usize) -> Array2<T>
where
    T: FftNum,
{
    let b = b.clone().remove_axis(Axis(1));
    let mut res = Array2::<T>::zeros((l, a.shape()[0]));
    for i in 0..l {
        let mut tmp = a.clone();
        for _ in 0..i {
            tmp = tmp.dot(a);
        }
        let out = c.dot(&tmp.dot(&b));
        res.slice_mut(s![i, ..]).assign(&out);
    }
    res
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
