/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

use ndarray::prelude::{Array, Array1, Axis, Dimension, NdFloat};
use ndarray::{concatenate, IntoDimension, RemoveAxis, ShapeError};
use num::cast::AsPrimitive;
use num::Float;

pub fn arange<T>(a: T, b: T, h: T) -> Array1<T>
where
    T: AsPrimitive<usize> + Float,
{
    let n: usize = ((b - a) / h).as_();
    let mut res = Array1::<T>::zeros(n);
    res[0] = a;
    for i in 1..n {
        res[i] = res[i - 1] + h;
    }
    res
}

pub fn cauchy_dot<T, D>(a: &Array<T, D>, lambda: &Array<T, D>, omega: &Array<T, D>) -> T
where
    D: Dimension,
    T: NdFloat,
{
    (a / (omega - lambda)).sum()
}

pub fn concat_iter<D, T>(axis: usize, iter: impl IntoIterator<Item = Array<T, D>>) -> Array<T, D>
where
    D: RemoveAxis,
    T: Clone,
{
    let mut arr = iter.into_iter().collect::<Vec<_>>();
    let mut out = arr.pop().unwrap();
    for i in arr {
        out = concatenate!(Axis(axis), out, i);
    }
    out
}

pub fn linarr<T, D>(dim: impl IntoDimension<Dim = D>) -> Result<Array<T, D>, ShapeError>
where
    D: Dimension,
    T: Float,
{
    let dim = dim.into_dimension();
    let n = dim.as_array_view().product();
    Array::linspace(T::one(), T::from(n).unwrap(), n).into_shape(dim)
}

pub fn now() -> u128 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis()
}
