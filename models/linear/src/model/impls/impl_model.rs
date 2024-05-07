/*
    Appellation: impl_model <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Config, Linear, LinearParams};
use concision::prelude::{Module, Predict, PredictError};
use nd::RemoveAxis;

impl<T, D> Module for Linear<T, D>
where
    D: RemoveAxis,
{
    type Config = Config;
    type Params = LinearParams<T, D>;

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

impl<X, Y, A, D> Predict<X> for Linear<A, D>
where
    D: RemoveAxis,
    LinearParams<A, D>: Predict<X, Output = Y>,
{
    type Output = Y;

    #[cfg_attr(
        feature = "tracing",
        tracing::instrument(skip_all, fields(name=%self.config.name), level = "debug", name = "predict", target = "linear")
    )]
    fn predict(&self, input: &X) -> Result<Self::Output, PredictError> {
        #[cfg(feature = "tracing")]
        tracing::debug!("Predicting with linear model");
        self.params.predict(input)
    }
}
