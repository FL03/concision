/*
    appellation: impl_model_params <module>
    authors: @FL03
*/
use crate::model::{ModelFeatures, ModelParamsBase};

use cnc::params::ParamsBase;
use ndarray::{DataOwned, Dimension, RawData};
use num_traits::{One, Zero};

impl<A, S> ModelParamsBase<S>
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

impl<A, S, D> core::ops::Index<usize> for ModelParamsBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: ndarray::Data<Elem = A>,
{
    type Output = ParamsBase<S, D>;

    fn index(&self, index: usize) -> &Self::Output {
        if index == 0 {
            &self.input
        } else if index == self.count_hidden() + 1 {
            &self.output
        } else {
            &self.hidden[index - 1]
        }
    }
}

impl<A, S, D> core::ops::IndexMut<usize> for ModelParamsBase<S, D>
where
    A: Clone,
    D: Dimension,
    S: ndarray::Data<Elem = A>,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index == 0 {
            &mut self.input
        } else if index == self.count_hidden() + 1 {
            &mut self.output
        } else {
            &mut self.hidden[index - 1]
        }
    }
}
