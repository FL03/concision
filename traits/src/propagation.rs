/*
    Appellation: predict <module>
    Contrib: @FL03
*/

/// The [`PropagationError`] type defines custom errors that can occur during forward and
/// backward propagation.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PropagationError {
    #[error("Forward Propagation Error: {0}")]
    ForwardError(&'static str),
    #[error("Backward Propagation Error: {0}")]
    BackwardError(&'static str),
    #[error("Mismatched Dimensions")]
    MismatchedDimensions,
    #[error("Invalid Input")]
    InvalidInput,
}

/// The [`Backward`] trait establishes a common interface for completing a single backward
/// step in a neural network or machine learning model.
pub trait Backward<X, Delta = X> {
    type Elem;

    fn backward(&mut self, input: &X, delta: &Delta, gamma: Self::Elem);
}

pub trait BackwardStep<T> {
    type Data<_X>;
    type Grad<_X>;
    type Output;

    fn backward(
        &mut self,
        input: &Self::Data<T>,
        delta: &Self::Grad<T>,
        gamma: T,
    ) -> Option<Self::Output>;
}

/// The [`Forward`] trait describes a common interface for objects designated to perform a
/// single forward step in a neural network or machine learning model.
pub trait Forward<Rhs> {
    type Output;
    /// a single forward step
    fn forward(&self, input: &Rhs) -> Option<Self::Output>;
    /// this method enables the forward pass to be generically _activated_ using some closure.
    /// This is useful for isolating the logic of the forward pass from that of the activation
    /// function and is often used by layers and models.
    fn forward_then<F>(&self, input: &Rhs, then: F) -> Option<Self::Output>
    where
        F: FnOnce(Self::Output) -> Self::Output,
    {
        self.forward(input).map(then)
    }
}

/*
 ************* Implementations *************
*/

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

impl<X, Y, A, S, D> Forward<X> for ArrayBase<S, D, A>
where
    A: Clone,
    D: Dimension,
    S: Data<Elem = A>,
    for<'a> X: Dot<ArrayBase<S, D, A>, Output = Y>,
{
    type Output = Y;

    fn forward(&self, input: &X) -> Option<Self::Output> {
        Some(input.dot(self))
    }
}
