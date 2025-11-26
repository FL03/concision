/*
    Appellation: reshape <module>
    Contrib: @FL03
*/
/// The [`DecrementAxis`] trait defines a method enabling an axis to decrement itself,
pub trait DecrementAxis {
    type Output;

    fn dec(&self) -> Self::Output;
}
/// The [`IncrementAxis`] trait defines a method enabling an axis to increment itself,
/// effectively adding a new axis to the array.
pub trait IncrementAxis {
    type Output;

    fn inc(&self) -> Self::Output;
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
use ndarray::{ArrayBase, Axis, Dimension, RawData, RawDataClone, RemoveAxis};

impl<D> DecrementAxis for D
where
    D: RemoveAxis,
{
    type Output = D::Smaller;

    fn dec(&self) -> Self::Output {
        self.remove_axis(Axis(self.ndim() - 1))
    }
}

impl<A, S, D> Unsqueeze for ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Output = ArrayBase<S, D::Larger>;

    fn unsqueeze(self, axis: usize) -> Self::Output {
        self.insert_axis(Axis(axis))
    }
}

impl<A, S, D> Unsqueeze for &ArrayBase<S, D, A>
where
    D: Dimension,
    S: RawDataClone<Elem = A>,
{
    type Output = ArrayBase<S, D::Larger>;

    fn unsqueeze(self, axis: usize) -> Self::Output {
        self.clone().insert_axis(Axis(axis))
    }
}
