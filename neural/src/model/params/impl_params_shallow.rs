/*
    Appellation: controller <module>
    Contrib: @FL03
*/
use crate::model::params::{ModelParamsBase, ShallowParamsBase};

use crate::model::ModelFeatures;
use crate::traits::ShallowNeuralStore;
use cnc::{ParamsBase, ReLU, Sigmoid};
use ndarray::{
    Array1, ArrayBase, Data, DataOwned, Dimension, Ix2, RawData, RemoveAxis, ScalarOperand,
};
use num_traits::Float;

impl<S, D, H, A> ModelParamsBase<S, D, H>
where
    D: Dimension,
    S: RawData<Elem = A>,
    H: ShallowNeuralStore<S, D>,
{
    /// create a new instance of the [`ModelParamsBase`] instance
    pub const fn shallow(input: ParamsBase<S, D>, hidden: H, output: ParamsBase<S, D>) -> Self {
        Self {
            input,
            hidden,
            output,
        }
    }
}

impl<S, D, A> ShallowParamsBase<S, D>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    #[allow(clippy::should_implement_trait)]
    /// initialize a new instance of the [`ShallowParamsBase`] with the given input, hidden,
    /// and output dimensions using the default values for the parameters
    pub fn default(input: D, hidden: D, output: D) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
        D: RemoveAxis,
    {
        Self {
            hidden: ParamsBase::default(hidden),
            input: ParamsBase::default(input),
            output: ParamsBase::default(output),
        }
    }
    /// returns the total number parameters within the model, including the input and output layers
    #[inline]
    pub fn size(&self) -> usize {
        let mut size = self.input().count_weight();
        size += self.hidden().count_weight();
        size + self.output().count_weight()
    }
    /// returns an immutable reference to the hidden weights
    pub const fn hidden_weights(&self) -> &ArrayBase<S, D> {
        self.hidden().weights()
    }
    /// returns an mutable reference to the hidden weights
    pub const fn hidden_weights_mut(&mut self) -> &mut ArrayBase<S, D> {
        self.hidden_mut().weights_mut()
    }
}

impl<S, A> ShallowParamsBase<S, Ix2>
where
    S: RawData<Elem = A>,
{
    pub fn from_features(features: ModelFeatures) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
    {
        Self {
            hidden: ParamsBase::default(features.dim_hidden()),
            input: ParamsBase::default(features.dim_input()),
            output: ParamsBase::default(features.dim_output()),
        }
    }
    /// forward input through the controller network
    pub fn forward(&self, input: &Array1<A>) -> cnc::Result<Array1<A>>
    where
        A: Float + ScalarOperand,
        S: Data,
    {
        // forward the input through the input layer; activate using relu
        let mut output = self.input().forward(input)?.relu();
        // forward the input through the hidden layer(s); activate using relu
        output = self.hidden().forward(&output)?.relu();
        // forward the input through the output layer; activate using sigmoid
        output = self.output().forward(&output)?.sigmoid();

        Ok(output)
    }
}

impl<A, S> Default for ShallowParamsBase<S, Ix2>
where
    S: DataOwned<Elem = A>,
    A: Clone + Default,
{
    fn default() -> Self {
        Self::from_features(ModelFeatures::default())
    }
}
