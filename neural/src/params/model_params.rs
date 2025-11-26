/*
    Appellation: store <module>
    Contrib: @FL03
*/

use cnc::params::ParamsBase;
use ndarray::{ArrayBase, Dimension, RawData};

use crate::{DeepModelRepr, RawHidden};

/// The [`ModelParamsBase`] object is a generic container for storing the parameters of a
/// neural network, regardless of the layout (e.g. shallow or deep). This is made possible
/// through the introduction of a generic hidden layer type, `H`, that allows us to define
/// aliases and additional traits for contraining the hidden layer type. That being said, we
/// don't reccoment using this type directly, but rather use the provided type aliases such as
/// [`DeepModelParams`] or [`ShallowModelParams`] or their owned variants. These provide a much
/// more straighforward interface for typing the parameters of a neural network. We aren't too
/// worried about the transmutation between the two since users desiring this ability should
/// simply stick with a _deep_ representation, initializing only a single layer within the
/// respective container.
///
/// This type also enables us to define a set of common initialization routines and introduce
/// other standards for dealing with parameters in a neural network.
pub struct ModelParamsBase<S, D, H, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = A>,
    H: RawHidden<S, D>,
{
    /// the input layer of the model
    pub(crate) input: ParamsBase<S, D>,
    /// a sequential stack of params for the model's hidden layers
    pub(crate) hidden: H,
    /// the output layer of the model
    pub(crate) output: ParamsBase<S, D>,
}
/// The base implementation for the [`ModelParamsBase`] type, which is generic over the
/// storage type `S`, the dimension `D`, and the hidden layer type `H`. This implementation
/// focuses on providing basic initialization routines and accessors for the various layers
/// within the model.
impl<S, D, H, A> ModelParamsBase<S, D, H, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
    H: RawHidden<S, D>,
{
    /// create a new instance of the [`ModelParamsBase`] instance
    pub const fn new(input: ParamsBase<S, D>, hidden: H, output: ParamsBase<S, D>) -> Self {
        Self {
            input,
            hidden,
            output,
        }
    }
    /// returns an immutable reference to the input layer of the model
    pub const fn input(&self) -> &ParamsBase<S, D> {
        &self.input
    }
    /// returns a mutable reference to the input layer of the model
    pub const fn input_mut(&mut self) -> &mut ParamsBase<S, D> {
        &mut self.input
    }
    /// returns an immutable reference to the hidden layers of the model
    pub const fn hidden(&self) -> &H {
        &self.hidden
    }
    /// returns a mutable reference to the hidden layers of the model
    pub const fn hidden_mut(&mut self) -> &mut H {
        &mut self.hidden
    }
    /// returns an immutable reference to the output layer of the model
    pub const fn output(&self) -> &ParamsBase<S, D> {
        &self.output
    }
    /// returns a mutable reference to the output layer of the model
    pub const fn output_mut(&mut self) -> &mut ParamsBase<S, D> {
        &mut self.output
    }
    /// set the input layer of the model
    #[inline]
    pub fn set_input(&mut self, input: ParamsBase<S, D>) -> &mut Self {
        *self.input_mut() = input;
        self
    }
    /// set the hidden layers of the model
    #[inline]
    pub fn set_hidden(&mut self, hidden: H) -> &mut Self {
        *self.hidden_mut() = hidden;
        self
    }
    /// set the output layer of the model
    #[inline]
    pub fn set_output(&mut self, output: ParamsBase<S, D>) -> &mut Self {
        *self.output_mut() = output;
        self
    }
    /// consumes the current instance and returns another with the specified input layer
    #[inline]
    pub fn with_input(self, input: ParamsBase<S, D>) -> Self {
        Self { input, ..self }
    }
    /// consumes the current instance and returns another with the specified hidden
    /// layer(s)
    #[inline]
    pub fn with_hidden(self, hidden: H) -> Self {
        Self { hidden, ..self }
    }
    /// consumes the current instance and returns another with the specified output layer
    #[inline]
    pub fn with_output(self, output: ParamsBase<S, D>) -> Self {
        Self { output, ..self }
    }
    /// returns an immutable reference to the hidden layers of the model as a slice
    #[inline]
    pub fn hidden_as_slice(&self) -> &[ParamsBase<S, D>]
    where
        H: DeepModelRepr<S, D>,
    {
        self.hidden().as_slice()
    }
    /// returns an immutable reference to the input bias
    pub const fn input_bias(&self) -> &ArrayBase<S, D::Smaller, A> {
        self.input().bias()
    }
    /// returns a mutable reference to the input bias
    pub const fn input_bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller, A> {
        self.input_mut().bias_mut()
    }
    /// returns an immutable reference to the input weights
    pub const fn input_weights(&self) -> &ArrayBase<S, D, A> {
        self.input().weights()
    }
    /// returns an mutable reference to the input weights
    pub const fn input_weights_mut(&mut self) -> &mut ArrayBase<S, D, A> {
        self.input_mut().weights_mut()
    }
    /// returns an immutable reference to the output bias
    pub const fn output_bias(&self) -> &ArrayBase<S, D::Smaller, A> {
        self.output().bias()
    }
    /// returns a mutable reference to the output bias
    pub const fn output_bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller, A> {
        self.output_mut().bias_mut()
    }
    /// returns an immutable reference to the output weights
    pub const fn output_weights(&self) -> &ArrayBase<S, D, A> {
        self.output().weights()
    }
    /// returns an mutable reference to the output weights
    pub const fn output_weights_mut(&mut self) -> &mut ArrayBase<S, D, A> {
        self.output_mut().weights_mut()
    }
    /// returns the number of hidden layers in the model
    pub fn count_hidden(&self) -> usize {
        self.hidden().count()
    }
    /// returns true if the stack is shallow; a neural network is considered to be _shallow_ if
    /// it has at most one hidden layer (`n <= 1`).
    #[inline]
    pub fn is_shallow(&self) -> bool {
        self.count_hidden() <= 1
    }
    /// returns true if the model stack of parameters is considered to be _deep_, meaning that
    /// there the number of hidden layers is greater than one.
    #[inline]
    pub fn is_deep(&self) -> bool {
        self.count_hidden() > 1
    }
    /// returns the total number of layers within the model, including the input and output layers
    #[inline]
    pub fn len(&self) -> usize {
        self.count_hidden() + 2 // +2 for input and output layers
    }
}
