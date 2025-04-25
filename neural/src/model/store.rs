/*
    Appellation: store <module>
    Contrib: @FL03
*/
use crate::ModelFeatures;
use cnc::params::ParamsBase;
use ndarray::{Dimension, Ix2, RawData};

pub type ModelParams<A = f64, D = Ix2> = ModelParamsBase<ndarray::OwnedRepr<A>, D>;

/// This object is an abstraction over the parameters of a deep neural network model. This is
/// done to isolate the necessary parameters from the specific logic within a model allowing us
/// to easily create additional stores for tracking velocities, gradients, and other metrics
/// we may need.
///
/// Additionally, this provides us with a way to introduce common creation routines for
/// initializing neural networks.
pub struct ModelParamsBase<S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) input: ParamsBase<S, D>,
    pub(crate) hidden: Vec<ParamsBase<S, D>>,
    pub(crate) output: ParamsBase<S, D>,
}

impl<A, S, D> ModelParamsBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn new(
        input: ParamsBase<S, D>,
        hidden: Vec<ParamsBase<S, D>>,
        output: ParamsBase<S, D>,
    ) -> Self {
        Self {
            input,
            hidden,
            output,
        }
    }
    /// returns true if the stack is shallow
    pub fn is_shallow(&self) -> bool {
        self.hidden.is_empty() || self.hidden.len() == 1
    }
    /// returns an immutable reference to the input layer of the model
    pub const fn input(&self) -> &ParamsBase<S, D> {
        &self.input
    }
    /// returns a mutable reference to the input layer of the model
    #[inline]
    pub fn input_mut(&mut self) -> &mut ParamsBase<S, D> {
        &mut self.input
    }
    /// returns an immutable reference to the hidden layers of the model
    pub const fn hidden(&self) -> &Vec<ParamsBase<S, D>> {
        &self.hidden
    }
    /// returns an immutable reference to the hidden layers of the model as a slice
    #[inline]
    pub fn hidden_as_slice(&self) -> &[ParamsBase<S, D>] {
        self.hidden.as_slice()
    }
    /// returns a mutable reference to the hidden layers of the model
    #[inline]
    pub fn hidden_mut(&mut self) -> &mut Vec<ParamsBase<S, D>> {
        &mut self.hidden
    }
    /// returns an immutable reference to the output layer of the model
    pub const fn output(&self) -> &ParamsBase<S, D> {
        &self.output
    }
    /// returns a mutable reference to the output layer of the model
    #[inline]
    pub fn output_mut(&mut self) -> &mut ParamsBase<S, D> {
        &mut self.output
    }
    /// set the input layer of the model
    pub fn set_input(&mut self, input: ParamsBase<S, D>) {
        self.input = input;
    }
    /// set the hidden layers of the model
    pub fn set_hidden<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = ParamsBase<S, D>>,
    {
        self.hidden = Vec::from_iter(iter);
    }
    /// set the output layer of the model
    pub fn set_output(&mut self, output: ParamsBase<S, D>) {
        self.output = output;
    }
    /// consumes the current instance and returns another with the specified input layer
    pub fn with_input(self, input: ParamsBase<S, D>) -> Self {
        Self { input, ..self }
    }
    /// consumes the current instance and returns another with the specified hidden layers
    pub fn with_hidden<I>(self, iter: I) -> Self
    where
        I: IntoIterator<Item = ParamsBase<S, D>>,
    {
        Self {
            hidden: Vec::from_iter(iter),
            ..self
        }
    }
    /// consumes the current instance and returns another with the specified output layer
    pub fn with_output(self, output: ParamsBase<S, D>) -> Self {
        Self { output, ..self }
    }
    /// returns the dimension of the input layer
    pub fn dim_input(&self) -> <D as Dimension>::Pattern {
        self.input().dim()
    }
    /// returns the dimension of the hidden layers
    pub fn dim_hidden(&self) -> <D as Dimension>::Pattern {
        assert!(self.hidden.iter().all(|p| p.dim() == self.hidden[0].dim()));
        self.hidden()[0].dim()
    }
    /// returns the dimension of the output layer
    pub fn dim_output(&self) -> <D as Dimension>::Pattern {
        self.output.dim()
    }
}

impl<A, S> ModelParamsBase<S>
where
    S: RawData<Elem = A>,
{
    /// create a new instance of the model;
    /// all parameters are initialized to their defaults (i.e., zero)
    pub fn default(features: ModelFeatures) -> Self
    where
        A: Clone + Default,
        S: ndarray::DataOwned,
    {
        let input = ParamsBase::default(features.d_input());
        let hidden = (0..features.layers())
            .map(|_| ParamsBase::default(features.d_hidden()))
            .collect::<Vec<_>>();
        let output = ParamsBase::default(features.d_output());
        Self::new(input, hidden, output)
    }
    /// create a new instance of the model;
    /// all parameters are initialized to zero
    pub fn ones(features: ModelFeatures) -> Self
    where
        A: Clone + num_traits::One,
        S: ndarray::DataOwned,
    {
        let input = ParamsBase::ones(features.d_input());
        let hidden = (0..features.layers())
            .map(|_| ParamsBase::ones(features.d_hidden()))
            .collect::<Vec<_>>();
        let output = ParamsBase::ones(features.d_output());
        Self::new(input, hidden, output)
    }
    /// create a new instance of the model;
    /// all parameters are initialized to zero
    pub fn zeros(features: ModelFeatures) -> Self
    where
        A: Clone + num_traits::Zero,
        S: ndarray::DataOwned,
    {
        let input = ParamsBase::zeros(features.d_input());
        let hidden = (0..features.layers())
            .map(|_| ParamsBase::zeros(features.d_hidden()))
            .collect::<Vec<_>>();
        let output = ParamsBase::zeros(features.d_output());
        Self::new(input, hidden, output)
    }

    pub fn forward<X, Y>(&self, input: &X) -> cnc::Result<Y>
    where
        A: Clone,
        S: ndarray::Data,
        ParamsBase<S, Ix2>: cnc::Forward<X, Output = Y> + cnc::Forward<Y, Output = Y>,
    {
        let mut output = self.input.forward(input)?;
        for layer in &self.hidden {
            output = layer.forward(&output)?;
        }
        self.output.forward(&output)
    }
}

impl<A, S, D> Clone for ModelParamsBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: ndarray::RawDataClone<Elem = A>,
{
    fn clone(&self) -> Self {
        Self {
            input: self.input.clone(),
            hidden: self.hidden.iter().map(|p| p.clone()).collect(),
            output: self.output.clone(),
        }
    }
}

impl<A, S, D> core::fmt::Debug for ModelParamsBase<S, D>
where
    A: core::fmt::Debug,
    D: Dimension,
    S: ndarray::Data<Elem = A>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ModelParams")
            .field("input", &self.input)
            .field("hidden", &self.hidden)
            .field("output", &self.output)
            .finish()
    }
}

impl<A, S, D> core::fmt::Display for ModelParamsBase<S, D>
where
    A: core::fmt::Debug,
    D: Dimension,
    S: ndarray::Data<Elem = A>,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{{ input: {:?}, hidden: {:?}, output: {:?} }}",
            self.input, self.hidden, self.output
        )
    }
}
