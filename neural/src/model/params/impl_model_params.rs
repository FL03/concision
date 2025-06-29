/*
    appellation: impl_model_params <module>
    authors: @FL03
*/
use crate::model::params::ModelParamsBase;

use crate::{DeepNeuralStore, RawHidden};
use cnc::params::ParamsBase;
use ndarray::{ArrayBase, Data, Dimension, RawData, RawDataClone};

impl<S, D, H, A> ModelParamsBase<S, D, H>
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
        H: DeepNeuralStore<S, D>,
    {
        self.hidden().as_slice()
    }
    /// returns an immutable reference to the input bias
    pub const fn input_bias(&self) -> &ArrayBase<S, D::Smaller> {
        self.input().bias()
    }
    /// returns a mutable reference to the input bias
    pub const fn input_bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller> {
        self.input_mut().bias_mut()
    }
    /// returns an immutable reference to the input weights
    pub const fn input_weights(&self) -> &ArrayBase<S, D> {
        self.input().weights()
    }
    /// returns an mutable reference to the input weights
    pub const fn input_weights_mut(&mut self) -> &mut ArrayBase<S, D> {
        self.input_mut().weights_mut()
    }
    /// returns an immutable reference to the output bias
    pub const fn output_bias(&self) -> &ArrayBase<S, D::Smaller> {
        self.output().bias()
    }
    /// returns a mutable reference to the output bias
    pub const fn output_bias_mut(&mut self) -> &mut ArrayBase<S, D::Smaller> {
        self.output_mut().bias_mut()
    }
    /// returns an immutable reference to the output weights
    pub const fn output_weights(&self) -> &ArrayBase<S, D> {
        self.output().weights()
    }
    /// returns an mutable reference to the output weights
    pub const fn output_weights_mut(&mut self) -> &mut ArrayBase<S, D> {
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

impl<A, S, D, H> Clone for ModelParamsBase<S, D, H>
where
    D: Dimension,
    H: RawHidden<S, D> + Clone,
    S: RawDataClone<Elem = A>,
    A: Clone,
{
    fn clone(&self) -> Self {
        Self {
            input: self.input().clone(),
            hidden: self.hidden().clone(),
            output: self.output().clone(),
        }
    }
}

impl<A, S, D, H> core::fmt::Debug for ModelParamsBase<S, D, H>
where
    D: Dimension,
    H: RawHidden<S, D> + core::fmt::Debug,
    S: Data<Elem = A>,
    A: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ModelParams")
            .field("input", self.input())
            .field("hidden", self.hidden())
            .field("output", self.output())
            .finish()
    }
}

impl<A, S, D, H> core::fmt::Display for ModelParamsBase<S, D, H>
where
    D: Dimension,
    H: RawHidden<S, D> + core::fmt::Debug,
    S: Data<Elem = A>,
    A: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{{ input: {i}, hidden: {h:?}, output: {o} }}",
            i = self.input(),
            h = self.hidden(),
            o = self.output()
        )
    }
}

impl<A, S, D, H> core::ops::Index<usize> for ModelParamsBase<S, D, H>
where
    D: Dimension,
    S: Data<Elem = A>,
    H: DeepNeuralStore<S, D>,
    A: Clone,
{
    type Output = ParamsBase<S, D>;

    fn index(&self, index: usize) -> &Self::Output {
        if index == 0 {
            self.input()
        } else if index == self.count_hidden() + 1 {
            self.output()
        } else {
            &self.hidden().as_slice()[index - 1]
        }
    }
}

impl<A, S, D, H> core::ops::IndexMut<usize> for ModelParamsBase<S, D, H>
where
    D: Dimension,
    S: Data<Elem = A>,
    H: DeepNeuralStore<S, D>,
    A: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index == 0 {
            self.input_mut()
        } else if index == self.count_hidden() + 1 {
            self.output_mut()
        } else {
            &mut self.hidden_mut().as_mut_slice()[index - 1]
        }
    }
}
