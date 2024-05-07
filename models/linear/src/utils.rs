/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::*;

pub(crate) fn build_bias<S, D, E, F>(biased: bool, dim: D, builder: F) -> Option<ArrayBase<S, E>>
where
    D: RemoveAxis<Smaller = E>,
    E: Dimension,
    F: Fn(E) -> ArrayBase<S, E>,
    S: RawData,
{
    let dim = bias_dim(dim);
    if biased {
        println!("Bias dimension: {:?}", &dim);
        Some(builder(dim))
    } else {
        None
    }
}

pub(crate) fn bias_dim<D, E>(dim: D) -> E
where
    D: RemoveAxis<Smaller = E>,
    E: Dimension,
{
    if dim.ndim() == 1 {
        dim.remove_axis(Axis(0))
    } else {
        dim.remove_axis(Axis(1))
    }
}
