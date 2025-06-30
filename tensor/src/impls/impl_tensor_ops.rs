/*
    appellation: impl_tensor_ops <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use crate::{Inverse, Tensor, TensorView, Transpose};

use ndarray::linalg::Dot;
use ndarray::{ArrayBase, Data, Dimension, Ix2, LinalgScalar, RawData, ScalarOperand};
use num_traits::NumAssign;

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// transpose the current tensor, returning a new instance with a view of the transposed data.
    pub fn transpose(&self) -> TensorView<'_, A, D>
    where
        S: Data,
    {
        TensorBase {
            store: self.store().t(),
        }
    }
}

impl<A> Inverse for Tensor<A, Ix2>
where
    A: Copy + NumAssign + ScalarOperand,
{
    type Output = Tensor<A, Ix2>;

    fn inverse(&self) -> Self::Output {
        let store = self.store().inverse().expect("Matrix is not invertible");
        TensorBase { store }
    }
}

impl<'a, A, S, D> Transpose for &'a TensorBase<S, D>
where
    A: 'a,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = TensorView<'a, A, D>;

    fn transpose(&self) -> Self::Output {
        let store = self.store().t();
        TensorBase { store }
    }
}

impl<A, S, D, X, S2, D2> Dot<X> for TensorBase<S, D>
where
    A: LinalgScalar,
    D: Dimension,
    D2: Dimension,
    S: RawData<Elem = A>,
    S2: RawData<Elem = A>,
    ArrayBase<S, D>: Dot<X, Output = ArrayBase<S2, D2>>,
{
    type Output = TensorBase<S2, D2>;

    fn dot(&self, rhs: &X) -> Self::Output {
        self.mapd(|store| Dot::dot(store, rhs))
    }
}
