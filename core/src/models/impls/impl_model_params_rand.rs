/*
    appellation: impl_model_params_rand <module>
    authors: @FL03
*/
use crate::models::{DeepParamsBase, ShallowParamsBase};

use crate::ModelFeatures;
use concision_init::distr as init;
use concision_init::{InitRand, rand_distr};
use concision_params::ParamsBase;
use ndarray::{DataOwned, Ix2};
use num_traits::{Float, FromPrimitive};

use rand_distr::uniform::{SampleUniform, Uniform};
use rand_distr::{Distribution, StandardNormal};

impl<A, S> ShallowParamsBase<S, Ix2, A>
where
    S: DataOwned<Elem = A>,
{
    /// consumes the controller to initialize the various parameters with random values
    pub fn init(self) -> Self
    where
        A: Float + num_traits::FromPrimitive + rand_distr::uniform::SampleUniform,
        rand_distr::StandardNormal: rand_distr::Distribution<A>,
    {
        // initialize the hidden layer(s)
        let hidden = ParamsBase::glorot_normal(self.hidden().dim());
        // Controller input weights (alphabet + state -> hidden)
        let input = ParamsBase::glorot_normal(self.input().dim());
        // initialize the output layers
        let output = ParamsBase::glorot_normal(self.output().dim());
        // return a new instance with the initialized layers
        Self {
            hidden,
            input,
            output,
        }
    }
    /// returns a new instance of the model initialized with the given features and random
    /// distribution
    pub fn init_rand<G, Ds>(features: ModelFeatures, distr: G) -> Self
    where
        G: Fn((usize, usize)) -> Ds,
        Ds: Clone + Distribution<A>,
        S: DataOwned,
    {
        Self {
            input: ParamsBase::rand(features.dim_input(), distr(features.dim_input())),
            hidden: ParamsBase::rand(features.dim_hidden(), distr(features.dim_hidden())),
            output: ParamsBase::rand(features.dim_output(), distr(features.dim_output())),
        }
    }
    /// initialize the model parameters using a glorot normal distribution
    pub fn glorot_normal(features: ModelFeatures) -> Self
    where
        A: Float + FromPrimitive,
        StandardNormal: Distribution<A>,
    {
        Self::init_rand(features, |(rows, cols)| init::XavierNormal::new(rows, cols))
    }
    /// initialize the model parameters using a glorot uniform distribution
    pub fn glorot_uniform(features: ModelFeatures) -> Self
    where
        A: Float + FromPrimitive + SampleUniform,
        <A as SampleUniform>::Sampler: Clone,
        Uniform<A>: Distribution<A>,
    {
        Self::init_rand(features, |(rows, cols)| {
            init::XavierUniform::new(rows, cols).expect("failed to create distribution")
        })
    }
}

impl<A, S> DeepParamsBase<S, Ix2, A>
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
        Self::init_rand(features, |(rows, cols)| init::XavierNormal::new(rows, cols))
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
