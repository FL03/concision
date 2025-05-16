/*
    Appellation: layout <module>
    Contrib: @FL03
*/
use super::ModelLayout;

/// The [`ModelFeatures`] provides a common way of defining the layout of a model. This is
/// used to define the number of input features, the number of hidden layers, the number of
/// hidden features, and the number of output features.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ModelFeatures {
    /// the number of input features
    pub(crate) input: usize,
    /// the dimension of hidden layers
    pub(crate) hidden: usize,
    /// the number of hidden layers
    pub(crate) layers: usize,
    /// the number of output features
    pub(crate) output: usize,
}

impl ModelFeatures {
    pub fn new(input: usize, hidden: usize, layers: usize, output: usize) -> Self {
        Self {
            input,
            hidden,
            layers,
            output,
        }
    }
    /// returns a copy of the input features for the model
    pub const fn input(&self) -> usize {
        self.input
    }
    /// returns a mutable reference to the input features for the model
    #[inline]
    pub const fn input_mut(&mut self) -> &mut usize {
        &mut self.input
    }
    /// returns a copy of the hidden features for the model
    pub const fn hidden(&self) -> usize {
        self.hidden
    }
    /// returns a mutable reference to the hidden features for the model
    #[inline]
    pub const fn hidden_mut(&mut self) -> &mut usize {
        &mut self.hidden
    }
    /// returns a copy of the number of hidden layers for the model
    pub const fn layers(&self) -> usize {
        self.layers
    }
    /// returns a mutable reference to the number of hidden layers for the model
    #[inline]
    pub const fn layers_mut(&mut self) -> &mut usize {
        &mut self.layers
    }
    /// returns a copy of the output features for the model
    pub const fn output(&self) -> usize {
        self.output
    }
    /// returns a mutable reference to the output features for the model
    #[inline]
    pub const fn output_mut(&mut self) -> &mut usize {
        &mut self.output
    }
    #[inline]
    /// sets the input features for the model
    pub fn set_input(&mut self, input: usize) -> &mut Self {
        self.input = input;
        self
    }
    #[inline]
    /// sets the hidden features for the model
    pub fn set_hidden(&mut self, hidden: usize) -> &mut Self {
        self.hidden = hidden;
        self
    }
    #[inline]
    /// sets the number of hidden layers for the model
    pub fn set_layers(&mut self, layers: usize) -> &mut Self {
        self.layers = layers;
        self
    }
    #[inline]
    /// sets the output features for the model
    pub fn set_output(&mut self, output: usize) -> &mut Self {
        self.output = output;
        self
    }
    /// consumes the current instance and returns a new instance with the given input
    pub fn with_input(self, input: usize) -> Self {
        Self { input, ..self }
    }

    /// consumes the current instance and returns a new instance with the given hidden
    /// features
    pub fn with_hidden(self, hidden: usize) -> Self {
        Self { hidden, ..self }
    }
    /// consumes the current instance and returns a new instance with the given number of
    /// hidden layers
    pub fn with_layers(self, layers: usize) -> Self {
        Self { layers, ..self }
    }
    /// consumes the current instance and returns a new instance with the given output
    /// features
    pub fn with_output(self, output: usize) -> Self {
        Self { output, ..self }
    }
    /// the dimension of the input layer; (input, hidden)
    pub fn dim_input(&self) -> (usize, usize) {
        (self.input(), self.hidden())
    }
    /// the dimension of the hidden layers; (hidden, hidden)
    pub fn dim_hidden(&self) -> (usize, usize) {
        (self.hidden(), self.hidden())
    }
    /// the dimension of the output layer; (hidden, output)
    pub fn dim_output(&self) -> (usize, usize) {
        (self.hidden(), self.output())
    }
    /// the total number of parameters in the model
    pub fn size(&self) -> usize {
        self.size_input() + self.size_hidden() + self.size_output()
    }
    /// the total number of input parameters in the model
    pub fn size_input(&self) -> usize {
        self.input() * self.hidden()
    }
    /// the total number of hidden parameters in the model
    pub fn size_hidden(&self) -> usize {
        self.hidden() * self.hidden() * self.layers()
    }
    /// the total number of output parameters in the model
    pub fn size_output(&self) -> usize {
        self.hidden() * self.output()
    }
}

impl ModelLayout for ModelFeatures {
    fn input(&self) -> usize {
        self.input()
    }
    fn input_mut(&mut self) -> &mut usize {
        self.input_mut()
    }
    fn hidden(&self) -> usize {
        self.hidden()
    }
    fn hidden_mut(&mut self) -> &mut usize {
        self.hidden_mut()
    }
    fn layers(&self) -> usize {
        self.layers()
    }
    fn layers_mut(&mut self) -> &mut usize {
        self.layers_mut()
    }
    fn output(&self) -> usize {
        self.output()
    }
    fn output_mut(&mut self) -> &mut usize {
        self.output_mut()
    }
}

impl Default for ModelFeatures {
    fn default() -> Self {
        Self {
            input: 16,
            hidden: 64,
            layers: 3,
            output: 16,
        }
    }
}

impl core::fmt::Display for ModelFeatures {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{{ input: {}, hidden: {}, layers: {}, output: {} }}",
            self.input, self.hidden, self.layers, self.output
        )
    }
}
