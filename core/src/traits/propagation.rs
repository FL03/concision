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

/// This trait defines the forward pass of the network

pub trait Forward<Rhs> {
    type Output;

    fn forward(&self, input: &Rhs) -> crate::Result<Self::Output>;
}
