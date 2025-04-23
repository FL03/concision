/*
    Appellation: mlp <module>
    Contrib: @FL03
*/
use crate::ModelFeatures;
use concision_core::params::Params;

/// This struct helps isolate the parameters of a deep neural network model providing common
/// creation and initialization routines
#[derive(Clone, Debug)]
pub struct ModelParams<A = f64> {
    pub(crate) input: Params<A>,
    pub(crate) hidden: Vec<Params<A>>,
    pub(crate) output: Params<A>,
}

impl<A> ModelParams<A> {
    /// create a new instance of the model;
    /// all parameters are initialized to their defaults (i.e., zero)
    pub fn default(features: ModelFeatures) -> Self
    where
        A: Clone + Default,
    {
        let input = Params::default(features.d_input());
        let hidden = (0..features.layers())
            .map(|_| Params::default(features.d_hidden()))
            .collect::<Vec<_>>();
        let output = Params::default(features.d_output());
        Self {
            input,
            hidden,
            output,
        }
    }
    /// create a new instance of the model;
    /// all parameters are initialized to zero
    pub fn ones(features: ModelFeatures) -> Self
    where
        A: Clone + num_traits::One,
    {
        let input = Params::ones(features.d_input());
        let hidden = (0..features.layers())
            .map(|_| Params::ones(features.d_hidden()))
            .collect::<Vec<_>>();
        let output = Params::ones(features.d_output());
        Self {
            input,
            hidden,
            output,
        }
    }
    /// create a new instance of the model;
    /// all parameters are initialized to zero
    pub fn zeros(features: ModelFeatures) -> Self
    where
        A: Clone + num_traits::Zero,
    {
        let input = Params::zeros(features.d_input());
        let hidden = (0..features.layers())
            .map(|_| Params::zeros(features.d_hidden()))
            .collect::<Vec<_>>();
        let output = Params::zeros(features.d_output());
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
    pub const fn input(&self) -> &Params<A> {
        &self.input
    }
    /// returns a mutable reference to the input layer of the model
    #[inline]
    pub fn input_mut(&mut self) -> &mut Params<A> {
        &mut self.input
    }
    /// returns an immutable reference to the hidden layers of the model
    pub const fn hidden(&self) -> &Vec<Params<A>> {
        &self.hidden
    }
    /// returns an immutable reference to the hidden layers of the model as a slice
    #[inline]
    pub fn hidden_as_slice(&self) -> &[Params<A>] {
        self.hidden.as_slice()
    }
    /// returns a mutable reference to the hidden layers of the model
    #[inline]
    pub fn hidden_mut(&mut self) -> &mut Vec<Params<A>> {
        &mut self.hidden
    }
    /// returns an immutable reference to the output layer of the model
    pub const fn output(&self) -> &Params<A> {
        &self.output
    }
    /// returns a mutable reference to the output layer of the model
    #[inline]
    pub fn output_mut(&mut self) -> &mut Params<A> {
        &mut self.output
    }
    /// set the input layer of the model
    pub fn set_input(&mut self, input: Params<A>) {
        self.input = input;
    }
    /// set the hidden layers of the model
    pub fn set_hidden<I>(&mut self, iter: I)
    where
        I: IntoIterator<Item = Params<A>>,
    {
        self.hidden = Vec::from_iter(iter);
    }
    /// set the output layer of the model
    pub fn set_output(&mut self, output: Params<A>) {
        self.output = output;
    }
    /// consumes the current instance and returns another with the specified input layer
    pub fn with_input(self, input: Params<A>) -> Self {
        Self { input, ..self }
    }
    /// consumes the current instance and returns another with the specified hidden layers
    pub fn with_hidden<I>(self, iter: I) -> Self
    where
        I: IntoIterator<Item = Params<A>>,
    {
        Self {
            hidden: Vec::from_iter(iter),
            ..self
        }
    }
    /// consumes the current instance and returns another with the specified output layer
    pub fn with_output(self, output: Params<A>) -> Self {
        Self { output, ..self }
    }

    pub fn dim_input(&self) -> (usize, usize) {
        self.input().dim()
    }

    pub fn dim_hidden(&self) -> (usize, usize) {
        assert!(self.hidden.iter().all(|p| p.dim() == self.hidden[0].dim()));
        self.hidden[0].dim()
    }

    pub fn dim_output(&self) -> (usize, usize) {
        self.output.dim()
    }
}
