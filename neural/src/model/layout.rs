#[doc(inline)]
pub use self::features::ModelFeatures;

mod features;

pub trait ModelLayout: Copy + core::fmt::Debug {
    /// returns a copy of the input features for the model
    fn input(&self) -> usize;
    /// returns a mutable reference to the input features for the model
    fn input_mut(&mut self) -> &mut usize;
    /// returns a copy of the hidden features for the model
    fn hidden(&self) -> usize;
    /// returns a mutable reference to the hidden features for the model
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
    /// sets the input features for the model
    fn set_input(&mut self, input: usize) -> &mut Self {
        *self.input_mut() = input;
        self
    }
    #[inline]
    /// sets the hidden features for the model
    fn set_hidden(&mut self, hidden: usize) -> &mut Self {
        *self.hidden_mut() = hidden;
        self
    }
    #[inline]
    /// sets the number of hidden layers for the model
    fn set_layers(&mut self, layers: usize) -> &mut Self {
        *self.layers_mut() = layers;
        self
    }
    #[inline]
    /// sets the output features for the model
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
