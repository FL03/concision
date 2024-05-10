/*
    Appellation: impl_model <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Config, Linear, LinearParams, ParamMode};
use concision::prelude::{Module, Predict, PredictError};
use nd::RemoveAxis;

impl<A, D, K> Module for Linear<K, A, D>
where
    D: RemoveAxis,
    K: ParamMode,
{
    type Config = Config<D, K>;
    type Params = LinearParams<K, A, D>;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn params(&self) -> &Self::Params {
        &self.params
    }

    fn params_mut(&mut self) -> &mut Self::Params {
        &mut self.params
    }
}

impl<U, V, K, A, D> Predict<U> for Linear<K, A, D>
where
    D: RemoveAxis,
    K: ParamMode,
    LinearParams<K, A, D>: Predict<U, Output = V>,
{
    type Output = V;

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, level = "debug", name = "predict", target = "linear")
    )]
    fn predict(&self, input: &U) -> Result<Self::Output, PredictError> {
        #[cfg(feature = "tracing")]
        tracing::debug!("Predicting with linear model");
        self.params().predict(input)
    }
}
