/*
    Appellation: impl_forward <module>
    Created At: 2025.12.14:09:36:14
    Contrib: @FL03
*/
use crate::{Forward, ForwardMut, ForwardOnce};

use ndarray::linalg::Dot;
use ndarray::{ArrayBase, Data, Dimension};

impl<F, X, Y> ForwardOnce<X> for F
where
    F: FnOnce(X) -> Y,
{
    type Output = Y;

    fn forward_once(self, input: X) -> Self::Output {
        self(input)
    }
}

impl<F, X, Y> Forward<X> for F
where
    F: Fn(&X) -> Y,
{
    type Output = Y;

    fn forward(&self, input: &X) -> Self::Output {
        self(input)
    }
}

impl<F, X, Y> ForwardMut<X> for F
where
    F: FnMut(&X) -> Y,
{
    type Output = Y;

    fn forward_mut(&mut self, input: &X) -> Self::Output {
        self(input)
    }
}

impl<X, Y, A, S, D> Forward<X> for ArrayBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    for<'a> X: Dot<ArrayBase<S, D, A>, Output = Y>,
{
    type Output = Y;

    fn forward(&self, input: &X) -> Self::Output {
        input.dot(self)
    }
}
