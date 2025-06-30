/*
    appellation: impl_tensor_repr <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use crate::tensor::{ArcTensor, Tensor, TensorView, TensorViewMut};
use ndarray::{Data, DataMut, Dimension, RawData};

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// returns a new tensor with the same shape and values, but with an _owned_ representation
    /// of the data.
    pub fn to_owned(&self) -> Tensor<A, D>
    where
        A: Clone,
        S: Data,
    {
        TensorBase {
            store: self.store().to_owned(),
        }
    }
    /// returns a new tensor with the same shape and values, but with an _shared_
    /// representation of the current data.
    pub fn to_shared(&self) -> ArcTensor<A, D>
    where
        A: Clone,
        S: Data,
    {
        TensorBase {
            store: self.store().to_shared(),
        }
    }
    /// returns a new tensor with the same shape and values, but with a _view_ of the current
    /// data.
    pub fn view(&self) -> TensorView<'_, A, D>
    where
        S: Data,
    {
        TensorBase {
            store: self.store().view(),
        }
    }
    /// returns a new tensor with the same shape and values, but with a mutable _view_ of the
    /// current data.
    pub fn view_mut(&mut self) -> TensorViewMut<'_, A, D>
    where
        S: DataMut,
    {
        TensorBase {
            store: self.store_mut().view_mut(),
        }
    }
}
