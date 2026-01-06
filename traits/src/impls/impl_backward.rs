/*
    Appellation: impl_backward <module>
    Created At: 2025.12.14:09:36:08
    Contrib: @FL03
*/
use crate::Backward;
use ndarray::linalg::Dot;
use ndarray::{Array, ArrayBase, ArrayView, Data, DataMut, Dimension};
use num_traits::Num;

impl<A, S, D, S1, D1, S2, D2> Backward<ArrayBase<S1, D1, A>, ArrayBase<S2, D2, A>>
    for ArrayBase<S, D, A>
where
    A: 'static + Copy + Num,
    D: Dimension,
    S: DataMut<Elem = A>,
    D1: Dimension,
    D2: Dimension,
    S1: Data<Elem = A>,
    S2: Data<Elem = A>,
    for<'b> &'b ArrayBase<S1, D1, A>: Dot<ArrayView<'b, A, D2>, Output = Array<A, D2>>,
{
    type Elem = A;

    fn backward(
        &mut self,
        input: &ArrayBase<S1, D1, A>,
        delta: &ArrayBase<S2, D2, A>,
        gamma: Self::Elem,
    ) {
        self.scaled_add(gamma, &input.dot(&delta.t()))
    }
}
