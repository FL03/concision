/*
    Appellation: traits <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Biased {
    type Bias;
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Self::Bias;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Self::Bias;
    /// Sets the bias of the layer.
    fn set_bias(&mut self, bias: Self::Bias);
}

pub trait Weighted {
    type Weight;
    /// Returns an owned reference to the weights of the layer.
    fn weights(&self) -> &Self::Weight;
    /// Returns a mutable reference to the weights of the layer.
    fn weights_mut(&mut self) -> &mut Self::Weight;
    /// Sets the weights of the layer.
    fn set_weights(&mut self, weights: Self::Weight);
}

pub trait StdParams: Biased + Weighted {}
