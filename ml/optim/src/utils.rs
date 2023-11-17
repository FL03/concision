/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::FTOL;
use ndarray::prelude::{Array, Array1, Dimension};
use num::Float;

pub fn minimize_inner<F>(
    w: &mut Array1<f64>,
    fg: F,
    epsilon: f64,
) -> anyhow::Result<(&mut Array1<f64>, f64, Array1<f64>)>
where
    F: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let (mut fp, mut gp) = fg(&w); // (cost, gradient)

    loop {
        w.scaled_add(-epsilon, &gp);
        let (f, g) = fg(&w);

        let exp = epsilon * norm_l2(&g);
        let delta = fp - f; // actual descrease; last - current
        if delta < exp * 0.5 {
            return Err(anyhow::anyhow!("Not enough decrease"));
        } else if delta < FTOL {
            return Ok((w, f, g));
        }
        fp = f;
        gp = g;
    }
}

pub fn minimize<F>(
    w: &mut Array1<f64>,
    fg: F,
    epsilon: f64,
    max_iter: usize,
) -> anyhow::Result<(&mut Array1<f64>, f64, Array1<f64>)>
where
    F: Fn(&Array1<f64>) -> (f64, Array1<f64>),
{
    let (mut fp, mut gp) = fg(&w); // (cost, gradient)

    for _ in 0..max_iter {
        w.scaled_add(-epsilon, &gp);
        let (f, g) = fg(&w);

        let exp = epsilon * norm_l2(&g);
        let delta = fp - f; // actual descrease; last - current
        if delta < exp * 0.5 {
            return Err(anyhow::anyhow!("Not enough decrease"));
        } else if delta < FTOL {
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
