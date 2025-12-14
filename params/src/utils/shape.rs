/*
    Appellation: shape <module>
    Created At: 2025.12.14:07:37:48
    Contrib: @FL03
*/
use ndarray::{Axis, LayoutRef, RemoveAxis};

/// A utilitarian function used to derive a valid bias shape from a given weight layout.
pub fn get_bias_shape<A, D>(layout: impl AsRef<LayoutRef<A, D>>) -> D::Smaller
where
    D: RemoveAxis,
{
    let layout = layout.as_ref();
    let dim = layout.raw_dim();
    dim.remove_axis(Axis(0))
}
