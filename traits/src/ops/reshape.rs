/*
    Appellation: reshape <module>
    Created At: 2025.11.26:13:10:09
    Contrib: @FL03
*/

/// The [`Unsqueeze`] trait establishes an interface for a routine that _unsqueezes_ an array,
/// by inserting a new axis at a specified position. This is useful for reshaping arrays to
/// meet specific dimensional requirements.
pub trait Unsqueeze {
    type Output;

    fn unsqueeze(self, axis: usize) -> Self::Output;
}

/// The [`DecrementAxis`] is used as a unary operator for removing a single axis
/// from a multidimensional array or tensor-like structure.
pub trait DecrementAxis {
    type Output;

    fn dec_axis(&self) -> Self::Output;
}

/// The [`IncrementAxis`] trait defines a method enabling an axis to increment itself,
/// effectively adding a new axis to the array.
pub trait IncrementAxis {
    type Output;

    fn inc_axis(self) -> Self::Output;
}

/*
 ************* Implementations *************
*/
use ndarray::{ArrayBase, Axis, Dimension, RawData, RawDataClone, RemoveAxis};

impl<D, E> DecrementAxis for D
where
    D: RemoveAxis<Smaller = E>,
    E: Dimension,
{
    type Output = E;

    fn dec_axis(&self) -> Self::Output {
        self.remove_axis(Axis(self.ndim() - 1))
    }
}

impl<D, E> IncrementAxis for D
where
    D: Dimension<Larger = E>,
    E: Dimension,
{
    type Output = E;

    fn inc_axis(self) -> Self::Output {
        self.insert_axis(Axis(self.ndim()))
    }
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
