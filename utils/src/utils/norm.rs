/*
    Appellation: norm <module>
    Contrib: @FL03
*/
use ndarray::{Array, ArrayBase, Axis, Data, Dimension, RemoveAxis};
use num_traits::{Float, FromPrimitive};

pub fn layer_norm<A, S, D>(x: &ArrayBase<S, D>, eps: f64) -> Array<A, D>
where
    A: Float + FromPrimitive,
    D: Dimension,
    S: Data<Elem = A>,
{
    let mean = x.mean().unwrap();
    let denom = {
        let eps = A::from(eps).unwrap();
        let var = x.var(A::zero());
        (var + eps).sqrt()
    };
    x.mapv(|xi| (xi - mean) / denom)
}

pub fn layer_norm_axis<A, S, D>(x: &ArrayBase<S, D>, axis: Axis, eps: f64) -> Array<A, D>
where
    A: Float + FromPrimitive,
    D: RemoveAxis,
    S: Data<Elem = A>,
{
    let eps = A::from(eps).unwrap();
    let mean = x.mean_axis(axis).unwrap();
    let var = x.var_axis(axis, A::zero());
    let inv_std = var.mapv(|v| (v + eps).recip().sqrt());
    
    (x - &mean) * &inv_std
}
