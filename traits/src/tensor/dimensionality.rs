/*
    Appellation: dimensionality <module>
    Created At: 2025.12.09:10:03:43
    Contrib: @FL03
*/

pub trait DimConst<const N: usize> {}

/// the [`Dim`] trait is used to define a type that can be used as a raw dimension.
/// This trait is primarily used to provide abstracted, generic interpretations of the
/// dimensions of the [`ndarray`] crate to ensure long-term compatibility.
pub trait Dim {
    type Shape;

    private! {}

    /// returns the rank of the dimension; the rank essentially speaks to the total number of
    /// axes defined by the dimension.
    fn rank(&self) -> usize;
    /// returns the total number of elements considered by the dimension
    fn size(&self) -> usize;
}

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

/*
 ************* Implementations *************
*/

impl<D> Dim for D
where
    D: ndarray::Dimension,
{
    type Shape = D::Pattern;

    seal! {}

    fn rank(&self) -> usize {
        self.ndim()
    }

    fn size(&self) -> usize {
        self.size()
    }
}
