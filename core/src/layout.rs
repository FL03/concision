/*
    appellation: layout <module>
    authors: @FL03
*/
mod impl_model_features;
mod impl_model_format;

/// The [`RawNetworkLayout`] trait defines a minimal interface for objects capable of representing
/// the _layout_; i.e. the number of input, hidden, and output features of
pub trait RawNetworkLayout {
    /// returns the total number of input features defined for the model
    fn input(&self) -> usize;
    /// returns the number of hidden features for the model
    fn hidden(&self) -> usize;
    /// returns the number of hidden layers within the network
    fn layers(&self) -> usize;
    /// the number of output features for the model
    fn output(&self) -> usize;
}

pub trait NetworkLayoutMut: RawNetworkLayout {
    /// returns a mutable reference to number of the input features for the model
    fn input_mut(&mut self) -> &mut usize;
    /// returns a mutable reference to the number of hidden features for the model
    fn hidden_mut(&mut self) -> &mut usize;
    /// returns a mutable reference to the number of hidden layers for the model
    fn layers_mut(&mut self) -> &mut usize;
    /// returns a mutable reference to the output features for the model
    fn output_mut(&mut self) -> &mut usize;
}

/// The [`ModelLayout`] trait defines an interface for object capable of representing the
/// _layout_; i.e. the number of input, hidden, and output features of a neural network model
/// containing some number of hidden layers.
pub trait NetworkLayout: RawNetworkLayout + NetworkLayoutMut + Clone + core::fmt::Debug {
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
}

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

/*
 ************* Implementations *************
*/

impl<T> RawNetworkLayout for &T
where
    T: RawNetworkLayout,
{
    fn input(&self) -> usize {
        <T as RawNetworkLayout>::input(self)
    }
    fn hidden(&self) -> usize {
        <T as RawNetworkLayout>::hidden(self)
    }
    fn layers(&self) -> usize {
        <T as RawNetworkLayout>::layers(self)
    }
    fn output(&self) -> usize {
        <T as RawNetworkLayout>::output(self)
    }
}

impl<T> RawNetworkLayout for &mut T
where
    T: RawNetworkLayout,
{
    fn input(&self) -> usize {
        <T as RawNetworkLayout>::input(self)
    }
    fn hidden(&self) -> usize {
        <T as RawNetworkLayout>::hidden(self)
    }
    fn layers(&self) -> usize {
        <T as RawNetworkLayout>::layers(self)
    }
    fn output(&self) -> usize {
        <T as RawNetworkLayout>::output(self)
    }
}

impl<T> NetworkLayout for T where T: NetworkLayoutMut + Copy + core::fmt::Debug {}

impl RawNetworkLayout for (usize, usize, usize, usize) {
    fn input(&self) -> usize {
        self.0
    }
    fn hidden(&self) -> usize {
        self.1
    }
    fn layers(&self) -> usize {
        self.2
    }
    fn output(&self) -> usize {
        self.3
    }
}

impl NetworkLayoutMut for (usize, usize, usize, usize) {
    fn input_mut(&mut self) -> &mut usize {
        &mut self.0
    }
    fn hidden_mut(&mut self) -> &mut usize {
        &mut self.1
    }
    fn layers_mut(&mut self) -> &mut usize {
        &mut self.2
    }
    fn output_mut(&mut self) -> &mut usize {
        &mut self.3
    }
}

impl RawNetworkLayout for [usize; 4] {
    fn input(&self) -> usize {
        self[0]
    }
    fn hidden(&self) -> usize {
        self[1]
    }
    fn layers(&self) -> usize {
        self[2]
    }
    fn output(&self) -> usize {
        self[3]
    }
}

impl NetworkLayoutMut for [usize; 4] {
    fn input_mut(&mut self) -> &mut usize {
        &mut self[0]
    }
    fn hidden_mut(&mut self) -> &mut usize {
        &mut self[1]
    }
    fn layers_mut(&mut self) -> &mut usize {
        &mut self[2]
    }
    fn output_mut(&mut self) -> &mut usize {
        &mut self[3]
    }
}
