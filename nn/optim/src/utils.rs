/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::FTOL;
use ndarray::prelude::{Array, Array1, Dimension};

use num::traits::Float;

pub fn minimize_inner<T, F>(
    w: &mut Array1<T>,
    fg: F,
    epsilon: T,
) -> anyhow::Result<(&mut Array1<T>, T, Array1<T>)>
where
    F: Fn(&Array1<T>) -> (T, Array1<T>),
    T: Float + 'static,
{
    let (mut fp, mut gp) = fg(&w); // (cost, gradient)

    loop {
        w.scaled_add(-epsilon, &gp);
        let (f, g) = fg(&w);

        let exp = epsilon * norm_l2(&g);
        let delta = fp - f; // actual descrease; last - current
        if delta < exp * T::from(2).unwrap().recip() {
            return Err(anyhow::anyhow!("Not enough decrease"));
        } else if delta < T::from(FTOL).unwrap() {
            return Ok((w, f, g));
        }
        fp = f;
        gp = g;
    }
}

pub fn minimize<T, F>(
    w: &mut Array1<T>,
    fg: F,
    epsilon: T,
    max_iter: usize,
) -> anyhow::Result<(&mut Array1<T>, T, Array1<T>)>
where
    F: Fn(&Array1<T>) -> (T, Array1<T>),
    T: Float + 'static,
{
    let (mut fp, mut gp) = fg(&w); // (cost, gradient)

    for _ in 0..max_iter {
        w.scaled_add(-epsilon, &gp);
        let (f, g) = fg(&w);

        let exp = epsilon * norm_l2(&g);
        let delta = fp - f; // actual descrease; last - current
        if delta < exp * T::from(2).unwrap().recip() {
            return Err(anyhow::anyhow!("Not enough decrease"));
        } else if delta < T::from(FTOL).unwrap() {
            return Ok((w, f, g));
        }
        fp = f;
        gp = g;
    }
    Ok((w, fp, gp))
}

pub fn norm_l2<T, D>(arr: &Array<T, D>) -> T
where
    D: Dimension,
    T: Float,
{
    arr.fold(T::zero(), |b, a| b + a.powi(2))
}
