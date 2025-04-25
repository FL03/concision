/*
    Appellation: predict <module>
    Contrib: @FL03
*/

/// A simple trait denoting a single backward pass through a layer of a neural network; the
/// trait
pub trait Backward<X, Y> {
    type HParam;
    type Output;

    fn backward(
        &mut self,
        input: &X,
        delta: &Y,
        gamma: Self::HParam,
    ) -> crate::Result<Self::Output>;
}

/// This trait denotes entities capable of performing a single forward step
pub trait Forward<Rhs> {
    type Output;
    /// a single forward step
    fn forward(&self, input: &Rhs) -> crate::Result<Self::Output>;
    /// this method enables the forward pass to be generically _activated_ using some closure.
    fn forward_then<F>(&self, input: &Rhs, then: F) -> crate::Result<Self::Output>
    where
        F: FnOnce(Self::Output) -> Self::Output,
    {
        self.forward(input).map(then)
    }
}
