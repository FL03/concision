/*
    appellation: impl_model_params <module>
    authors: @FL03
*/
use crate::model::{DeepParamsBase, ModelParamsBase};

use crate::model::ModelFeatures;
use crate::traits::DeepNeuralStore;
use cnc::params::ParamsBase;
use ndarray::{Data, DataOwned, Dimension, Ix2, RawData};
use num_traits::{One, Zero};



impl<S, D, H, A> ModelParamsBase<S, D, H>
where
    D: Dimension,
    S: RawData<Elem = A>,
    H: DeepNeuralStore<S, D>,
{
    /// create a new instance of the [`ModelParamsBase`] instance
    pub const fn deep(input: ParamsBase<S, D>, hidden: H, output: ParamsBase<S, D>) -> Self {
        Self {
            input,
            hidden,
            output,
        }
    }
}

impl<A, S, D> DeepParamsBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// returns the total number parameters within the model, including the input and output layers
    #[inline]
    pub fn size(&self) -> usize {
        let mut size = self.input().count_weight();
        for layer in self.hidden() {
            size += layer.count_weight();
        }
        size + self.output().count_weight()
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
}

impl<A, S> DeepParamsBase<S, Ix2>
where
    S: RawData<Elem = A>,
{
    /// create a new instance of the model;
    /// all parameters are initialized to their defaults (i.e., zero)
    pub fn default(features: ModelFeatures) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
    {
        let input = ParamsBase::default(features.dim_input());
        let hidden = (0..features.layers())
            .map(|_| ParamsBase::default(features.dim_hidden()))
            .collect::<Vec<_>>();
        let output = ParamsBase::default(features.dim_output());
        Self::new(input, hidden, output)
    }
    /// create a new instance of the model;
    /// all parameters are initialized to zero
    pub fn ones(features: ModelFeatures) -> Self
    where
        A: Clone + One,
        S: DataOwned,
    {
        let input = ParamsBase::ones(features.dim_input());
        let hidden = (0..features.layers())
            .map(|_| ParamsBase::ones(features.dim_hidden()))
            .collect::<Vec<_>>();
        let output = ParamsBase::ones(features.dim_output());
        Self::new(input, hidden, output)
    }
    /// create a new instance of the model;
    /// all parameters are initialized to zero
    pub fn zeros(features: ModelFeatures) -> Self
    where
        A: Clone + Zero,
        S: DataOwned,
    {
        let input = ParamsBase::zeros(features.dim_input());
        let hidden = (0..features.layers())
            .map(|_| ParamsBase::zeros(features.dim_hidden()))
            .collect::<Vec<_>>();
        let output = ParamsBase::zeros(features.dim_output());
        Self::new(input, hidden, output)
    }
}
