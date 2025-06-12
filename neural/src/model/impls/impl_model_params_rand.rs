/*
    appellation: impl_model_params_rand <module>
    authors: @FL03
*/

use crate::model::{ModelFeatures, ModelParamsBase};

use cnc::init::{self, Initialize};
use cnc::params::ParamsBase;
use cnc::rand_distr;

use ndarray::DataOwned;
use num_traits::{Float, FromPrimitive};
use rand_distr::uniform::{SampleUniform, Uniform};
use rand_distr::{Distribution, StandardNormal};

impl<A, S> ModelParamsBase<S>
where
    S: DataOwned<Elem = A>,
{
    /// returns a new instance of the model initialized with the given features and random
    /// distribution
    pub fn init_rand<G, Ds>(features: ModelFeatures, distr: G) -> Self
    where
        G: Fn((usize, usize)) -> Ds,
        Ds: Clone + Distribution<A>,
        S: DataOwned,
    {
        let input = ParamsBase::rand(features.dim_input(), distr(features.dim_input()));
        let hidden = (0..features.layers())
            .map(|_| ParamsBase::rand(features.dim_hidden(), distr(features.dim_hidden())))
            .collect::<Vec<_>>();

        let output = ParamsBase::rand(features.dim_output(), distr(features.dim_output()));

        Self::new(input, hidden, output)
    }
    /// initialize the model parameters using a glorot normal distribution
    pub fn glorot_normal(features: ModelFeatures) -> Self
    where
        A: Float + FromPrimitive,
        StandardNormal: Distribution<A>,
    {
        Self::init_rand(features, |(rows, cols)| {
            cnc::init::XavierNormal::new(rows, cols)
        })
    }
    /// initialize the model parameters using a glorot uniform distribution
    pub fn glorot_uniform(features: ModelFeatures) -> Self
    where
        A: Clone + Float + FromPrimitive + SampleUniform,
        <S::Elem as SampleUniform>::Sampler: Clone,
        Uniform<S::Elem>: Distribution<S::Elem>,
    {
        Self::init_rand(features, |(rows, cols)| {
            init::XavierUniform::new(rows, cols).expect("failed to create distribution")
        })
    }
}
