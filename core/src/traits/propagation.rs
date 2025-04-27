/*
    Appellation: predict <module>
    Contrib: @FL03
*/

/// [Backward] propagate a delta through the system;
pub trait Backward<X, Delta = X> {
    type Elem;
    type Output;

    fn backward(
        &mut self,
        input: &X,
        delta: &Delta,
        gamma: Self::Elem,
    ) -> crate::Result<Self::Output>;
}

/// This trait denotes entities capable of performing a single forward step
pub trait Forward<Rhs> {
    type Output;
    /// a single forward step
    fn forward(&self, input: &Rhs) -> crate::Result<Self::Output>;
    /// this method enables the forward pass to be generically _activated_ using some closure.
    /// This is useful for isolating the logic of the forward pass from that of the activation
    /// function and is often used by layers and models.
    fn forward_then<F>(&self, input: &Rhs, then: F) -> crate::Result<Self::Output>
    where
        F: FnOnce(Self::Output) -> Self::Output,
    {
        self.forward(input).map(then)
    }
}
