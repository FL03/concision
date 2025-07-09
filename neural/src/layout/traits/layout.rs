/*
    appellation: layout <module>
    authors: @FL03
*/

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
