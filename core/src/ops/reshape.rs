/*
    Appellation: reshape <module>
    Contrib: @FL03
*/
use ndarray::{ArrayBase, Axis, Dimension, RawData, RawDataClone, RemoveAxis};

/// This trait enables an array to remove an axis from itself
pub trait DecrementAxis {
    type Output;

    fn dec(&self) -> Self::Output;
}

pub trait IncrementAxis {
    type Output;

    fn inc(&self) -> Self::Output;
}

pub trait Unsqueeze {
    type Output;

    fn unsqueeze(self, axis: usize) -> Self::Output;
}
/*
 ************* Implementations *************
*/
impl<D> DecrementAxis for D
where
    D: RemoveAxis,
{
    type Output = D::Smaller;

    fn dec(&self) -> Self::Output {
        self.remove_axis(Axis(self.ndim() - 1))
    }
}

impl<A, S, D> Unsqueeze for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Output = ArrayBase<S, D::Larger>;

    fn unsqueeze(self, axis: usize) -> Self::Output {
        self.insert_axis(Axis(axis))
    }
}

impl<A, S, D> Unsqueeze for &ArrayBase<S, D>
where
    D: Dimension,
    S: RawDataClone<Elem = A>,
{
    type Output = ArrayBase<S, D::Larger>;

    fn unsqueeze(self, axis: usize) -> Self::Output {
        self.clone().insert_axis(Axis(axis))
    }
}
