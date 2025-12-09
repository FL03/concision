/*
    Appellation: shape <module>
    Created At: 2025.11.26:13:10:09
    Contrib: @FL03
*/

/// the [`Dim`] trait is used to define a type that can be used as a raw dimension.
/// This trait is primarily used to provide abstracted, generic interpretations of the
/// dimensions of the [`ndarray`] crate to ensure long-term compatibility.
pub trait RawDimension {
    private! {}
}

pub trait Dim: RawDimension {
    /// returns the total number of elements considered by the dimension
    fn size(&self) -> usize;
}


/// The [`Unsqueeze`] trait establishes an interface for a routine that _unsqueezes_ an array,
/// by inserting a new axis at a specified position. This is useful for reshaping arrays to
/// meet specific dimensional requirements.
pub trait Unsqueeze {
    type Output;

    fn unsqueeze(self, axis: usize) -> Self::Output;
}

/*
 ************* Implementations *************
*/
use ndarray::{ArrayBase, Axis, Dimension, RawData, RawDataClone};

impl<D> RawDimension for D
where
    D: ndarray::Dimension,
{
    seal! {}
}


impl<S, D, A> Unsqueeze for ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Output = ArrayBase<S, D::Larger>;

    fn unsqueeze(self, axis: usize) -> Self::Output {
        self.insert_axis(Axis(axis))
    }
}

impl<S, D, A> Unsqueeze for &ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawDataClone<Elem = A>,
{
    type Output = ArrayBase<S, D::Larger>;

    fn unsqueeze(self, axis: usize) -> Self::Output {
        self.clone().insert_axis(Axis(axis))
    }
}
