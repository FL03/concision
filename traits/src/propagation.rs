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
    type Output;

    fn backward(&mut self, input: &X, delta: &Delta, gamma: Self::Elem) -> Option<Self::Output>;
}

/// The [`Forward`] trait defines an interface that is used to perform a single forward step
/// within a neural network or machine learning model.
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
use ndarray::{ArrayBase, Data, Dimension, LinalgScalar};
use num_traits::FromPrimitive;

impl<X, Y, Dx, A, S, D> Backward<X, Y> for ArrayBase<S, D, A>
where
    A: LinalgScalar + FromPrimitive,
    D: Dimension,
    S: ndarray::DataMut<Elem = A>,
    Dx: core::ops::Mul<A, Output = Dx>,
    for<'a> X: Dot<Y, Output = Dx>,
    for<'a> &'a Self: core::ops::Add<&'a Dx, Output = Self>,
{
    type Elem = A;
    type Output = ();

    fn backward(&mut self, input: &X, delta: &Y, gamma: Self::Elem) -> Option<Self::Output> {
        let dx = input.dot(delta);
        let next = &*self + &(dx * gamma);
        self.assign(&next);
        Some(())
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
