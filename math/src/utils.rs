/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::linalg::Dot;
use ndarray::prelude::{Array, NdFloat};
use ndarray::Dimension;
use num::FromPrimitive;

pub fn covariance<T, D>(ddof: T, x: &Array<T, D>, y: &Array<T, D>) -> anyhow::Result<Array<T, D>>
where
    D: Dimension,
    T: Default + FromPrimitive + NdFloat,
    Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>>,
{
    let x_mean = x.mean().unwrap_or_default();
    let y_mean = y.mean().unwrap_or_default();
    let xs = x - x_mean;
    let ys = y - y_mean;
    let cov = xs.dot(&ys.t().to_owned());
    let scale = T::one() / (T::from(x.len()).unwrap() - ddof);
    Ok(cov * scale)
}
