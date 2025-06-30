/*
    appellation: impl_tensor <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use ndarray::{ArrayBase, Data, Dimension, RawData, RawDataClone};

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
}

impl<A, S, D> Clone for TensorBase<S, D>
where
    A: Clone,
    S: RawDataClone<Elem = A>,
    D: Dimension,
{
    fn clone(&self) -> Self {
        TensorBase {
            store: self.store().clone(),
        }
    }
}

impl<A, S, D> Copy for TensorBase<S, D>
where
    A: Copy,
    S: RawDataClone<Elem = A> + Copy,
    D: Dimension + Copy,
{
}

impl<A, S, D> PartialEq for TensorBase<S, D>
where
    A: PartialEq,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn eq(&self, other: &Self) -> bool {
        self.store() == other.store()
    }
}

impl<A, S, D> PartialEq<ArrayBase<S, D>> for TensorBase<S, D>
where
    A: PartialEq,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn eq(&self, other: &ArrayBase<S, D>) -> bool {
        self.store() == other
    }
}

impl<A, S, D> PartialEq<&ArrayBase<S, D>> for TensorBase<S, D>
where
    A: PartialEq,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn eq(&self, other: &&ArrayBase<S, D>) -> bool {
        self.store() == *other
    }
}

impl<A, S, D> PartialEq<&mut ArrayBase<S, D>> for TensorBase<S, D>
where
    A: PartialEq,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn eq(&self, other: &&mut ArrayBase<S, D>) -> bool {
        self.store() == *other
    }
}
