/*
    appellation: convert <module>
    authors: @FL03
*/
use ndarray::{Axis, Dimension, RemoveAxis};

/// The [`IntoAxis`] trait is used to define a conversion routine that takes a type and wraps
/// it in an [`Axis`] type.
pub trait IntoAxis {
    fn into_axis(self) -> Axis;
}

/// The [`AsBiasDim`] trait is used to define a type that can be used to get the bias dimension
/// of the parameters.
pub trait AsBiasDim<D: Dimension> {
    /// returns the bias dimension of the parameters
    fn as_bias_dim(&self) -> D;
}

/*
 ************* Implementations *************
*/

impl<S> IntoAxis for S
where
    S: AsRef<usize>,
{
    fn into_axis(self) -> Axis {
        Axis(*self.as_ref())
    }
}

impl<A, B> AsBiasDim<B> for A
where
    A: RemoveAxis<Smaller = B>,
    B: Dimension,
{
    /// returns the bias dimension of the parameters by removing the "zero-th" axis from the
    /// given dimension
    fn as_bias_dim(&self) -> B {
        self.remove_axis(Axis(0))
    }
}
