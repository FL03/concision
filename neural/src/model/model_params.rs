/*
    Appellation: store <module>
    Contrib: @FL03
*/
use cnc::params::ParamsBase;
use ndarray::{Data, Dimension, Ix2, RawData};

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
    /// the input layer of the model
    pub(crate) input: ParamsBase<S, D>,
    /// a sequential stack of params for the model's hidden layers
    pub(crate) hidden: Vec<ParamsBase<S, D>>,
    /// the output layer of the model
    pub(crate) output: ParamsBase<S, D>,
}

impl<A, S, D> ModelParamsBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// returns a new instance of the [`ModelParamsBase`] with the specified input, hidden, and
    /// output layers.
    pub const fn new(
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
    /// returns an immutable reference to the input layer of the model
    pub const fn input(&self) -> &ParamsBase<S, D> {
        &self.input
    }
    /// returns a mutable reference to the input layer of the model
    pub const fn input_mut(&mut self) -> &mut ParamsBase<S, D> {
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
    pub const fn hidden_mut(&mut self) -> &mut Vec<ParamsBase<S, D>> {
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
    pub fn set_hidden(&mut self, hidden: Vec<ParamsBase<S, D>>) -> &mut Self {
        *self.hidden_mut() = hidden;
        self
    }
    /// set the layer at the specified index in the hidden layers of the model
    ///
    /// ## Panics
    ///
    /// Panics if the index is out of bounds or if the dimension of the provided layer is
    /// inconsistent with the others in the stack.
    #[inline]
    pub fn set_hidden_layer(&mut self, idx: usize, layer: ParamsBase<S, D>) -> &mut Self {
        if layer.dim() != self.dim_hidden() {
            panic!(
                "the dimension of the layer ({:?}) does not match the dimension of the hidden layers ({:?})",
                layer.dim(),
                self.dim_hidden()
            );
        }
        self.hidden_mut()[idx] = layer;
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
    /// consumes the current instance and returns another with the specified hidden layers
    #[inline]
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
    #[inline]
    pub fn with_output(self, output: ParamsBase<S, D>) -> Self {
        Self { output, ..self }
    }
    /// returns the dimension of the input layer
    #[inline]
    pub fn dim_input(&self) -> <D as Dimension>::Pattern {
        self.input().dim()
    }
    /// returns the dimension of the hidden layers
    #[inline]
    pub fn dim_hidden(&self) -> <D as Dimension>::Pattern {
        // verify that all hidden layers have the same dimension
        assert!(
            self.hidden()
                .iter()
                .all(|p| p.dim() == self.hidden()[0].dim())
        );
        // use the first hidden layer's dimension as the representative
        // dimension for all hidden layers
        self.hidden()[0].dim()
    }
    /// returns the dimension of the output layer
    #[inline]
    pub fn dim_output(&self) -> <D as Dimension>::Pattern {
        self.output().dim()
    }
    /// returns the hidden layer associated with the given index
    #[inline]
    pub fn get_hidden_layer<I>(&self, idx: I) -> Option<&I::Output>
    where
        I: core::slice::SliceIndex<[ParamsBase<S, D>]>,
    {
        self.hidden().get(idx)
    }
    /// returns a mutable reference to the hidden layer associated with the given index
    #[inline]
    pub fn get_hidden_layer_mut<I>(&mut self, idx: I) -> Option<&mut I::Output>
    where
        I: core::slice::SliceIndex<[ParamsBase<S, D>]>,
    {
        self.hidden_mut().get_mut(idx)
    }
    /// sequentially forwards the input through the model without any activations or other
    /// complexities in-between. not overly usefuly, but it is here for completeness
    #[inline]
    pub fn forward<X, Y>(&self, input: &X) -> cnc::Result<Y>
    where
        A: Clone,
        S: Data,
        ParamsBase<S, D>: cnc::Forward<X, Output = Y> + cnc::Forward<Y, Output = Y>,
    {
        // forward the input through the input layer
        let mut output = self.input().forward(input)?;
        // forward the input through each of the hidden layers
        for layer in self.hidden() {
            output = layer.forward(&output)?;
        }
        // finally, forward the output through the output layer
        self.output().forward(&output)
    }
    /// returns true if the stack is shallow; a neural network is considered to be _shallow_ if
    /// it has at most one hidden layer (`n <= 1`).
    #[inline]
    pub fn is_shallow(&self) -> bool {
        self.count_hidden() <= 1 || self.hidden().is_empty()
    }
    /// returns true if the model stack of parameters is considered to be _deep_, meaning that
    /// there the number of hidden layers is greater than one.
    #[inline]
    pub fn is_deep(&self) -> bool {
        self.count_hidden() > 1
    }
    /// returns the total number of hidden layers within the model
    #[inline]
    pub fn count_hidden(&self) -> usize {
        self.hidden().len()
    }
    /// returns the total number of layers within the model, including the input and output layers
    #[inline]
    pub fn len(&self) -> usize {
        self.count_hidden() + 2 // +2 for input and output layers
    }
    /// returns the total number parameters within the model, including the input and output layers
    #[inline]
    pub fn size(&self) -> usize {
        let mut size = self.input().count_weight();
        for layer in self.hidden() {
            size += layer.count_weight();
        }
        size + self.output().count_weight()
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
            input: self.input().clone(),
            hidden: self.hidden().to_vec(),
            output: self.output().clone(),
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
