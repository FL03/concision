/*
    Appellation: propagate <module>
    Created At: 2026.01.06:14:13:38
    Contrib: @FL03
*/

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

    fn backward(&mut self, input: &Self::Data<T>, delta: &Self::Grad<T>, gamma: T) -> Self::Output;
}

/// A consuming implementation of forward propagation
pub trait ForwardOnce<Rhs> {
    type Output;
    /// a single forward step consuming the implementor
    fn forward_once(self, input: Rhs) -> Self::Output;
}
/// The [`Forward`] trait describes a common interface for objects designated to perform a
/// single forward step in a neural network or machine learning model.
pub trait Forward<Rhs> {
    type Output;
    /// a single forward step
    fn forward(&self, input: &Rhs) -> Self::Output;
    /// this method enables the forward pass to be generically _activated_ using some closure.
    /// This is useful for isolating the logic of the forward pass from that of the activation
    /// function and is often used by layers and models.
    fn forward_then<F>(&self, input: &Rhs, then: F) -> Self::Output
    where
        F: FnOnce(Self::Output) -> Self::Output,
    {
        then(self.forward(input))
    }
}

pub trait ForwardMut<Rhs> {
    type Output;
    /// a single forward step with mutable access
    fn forward_mut(&mut self, input: &Rhs) -> Self::Output;
}
