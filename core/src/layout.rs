/*
    appellation: layout <module>
    authors: @FL03
*/
mod impl_model_features;
mod impl_model_format;

/// The [`ModelFormat`] type enumerates the various formats a neural network may take, either
/// shallow or deep, providing a unified interface for accessing the number of hidden features
/// and layers in the model. This is done largely for simplicity, as it eliminates the need to
/// define a particular _type_ of network as its composition has little impact on the actual
/// requirements / algorithms used to train or evaluate the model (that is, outside of the
/// obvious need to account for additional hidden layers in deep configurations). In other
/// words, both shallow and deep networks are requried to implement the same traits and
/// fulfill the same requirements, so it makes sense to treat them as a single type with
/// different configurations. The differences between the networks are largely left to the
/// developer and their choice of activation functions, optimizers, and other considerations.
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, strum::EnumCount, strum::EnumIs,
)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ModelFormat {
    Shallow { hidden: usize },
    Deep { hidden: usize, layers: usize },
}

/// The [`ModelFeatures`] provides a common way of defining the layout of a model. This is
/// used to define the number of input features, the number of hidden layers, the number of
/// hidden features, and the number of output features.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ModelFeatures {
    /// the number of input features
    pub(crate) input: usize,
    /// the features of the "inner" layers
    pub(crate) inner: ModelFormat,
    /// the number of output features
    pub(crate) output: usize,
}


/// The [`ModelLayout`] trait defines an interface for object capable of representing the
/// _layout_; i.e. the number of input, hidden, and output features of a neural network model
/// containing some number of hidden layers.
pub trait ModelLayout: Copy + core::fmt::Debug {
    /// returns a copy of the number of input features for the model
    fn input(&self) -> usize;
    /// returns a mutable reference to number of the input features for the model
    fn input_mut(&mut self) -> &mut usize;
    /// returns a copy of the number of hidden features for the model
    fn hidden(&self) -> usize;
    /// returns a mutable reference to the number of hidden features for the model
    fn hidden_mut(&mut self) -> &mut usize;
    /// returns a copy of the number of hidden layers for the model
    fn layers(&self) -> usize;
    /// returns a mutable reference to the number of hidden layers for the model
    fn layers_mut(&mut self) -> &mut usize;
    /// returns a copy of the output features for the model
    fn output(&self) -> usize;
    /// returns a mutable reference to the output features for the model
    fn output_mut(&mut self) -> &mut usize;
    #[inline]
    /// update the number of input features for the model and return a mutable reference to the
    /// current layout.
    fn set_input(&mut self, input: usize) -> &mut Self {
        *self.input_mut() = input;
        self
    }
    #[inline]
    /// update the number of hidden features for the model and return a mutable reference to
    /// the current layout.
    fn set_hidden(&mut self, hidden: usize) -> &mut Self {
        *self.hidden_mut() = hidden;
        self
    }
    #[inline]
    /// update the number of hidden layers for the model and return a mutable reference to
    /// the current layout.
    fn set_layers(&mut self, layers: usize) -> &mut Self {
        *self.layers_mut() = layers;
        self
    }
    #[inline]
    /// update the number of output features for the model and return a mutable reference to
    /// the current layout.
    fn set_output(&mut self, output: usize) -> &mut Self {
        *self.output_mut() = output;
        self
    }
    /// the dimension of the input layer; (input, hidden)
    fn dim_input(&self) -> (usize, usize) {
        (self.input(), self.hidden())
    }
    /// the dimension of the hidden layers; (hidden, hidden)
    fn dim_hidden(&self) -> (usize, usize) {
        (self.hidden(), self.hidden())
    }
    /// the dimension of the output layer; (hidden, output)
    fn dim_output(&self) -> (usize, usize) {
        (self.hidden(), self.output())
    }
    /// the total number of parameters in the model
    fn size(&self) -> usize {
        self.size_input() + self.size_hidden() + self.size_output()
    }
    /// the total number of input parameters in the model
    fn size_input(&self) -> usize {
        self.input() * self.hidden()
    }
    /// the total number of hidden parameters in the model
    fn size_hidden(&self) -> usize {
        self.hidden() * self.hidden() * self.layers()
    }
    /// the total number of output parameters in the model
    fn size_output(&self) -> usize {
        self.hidden() * self.output()
    }
}