/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::prelude::{DropoutLayer, Forward, Predict, PredictError, ReLU};
use linear::{Biased, Linear, ParamMode};
use nd::prelude::*;
use nd::{RemoveAxis, ScalarOperand};
use num::traits::Num;

// 
pub struct FeedForwardNetwork<A = f64, D = Ix2, K = Biased>
where
    D: Dimension,
{
    #[cfg(feature = "rand")]
    pub(crate) dropout: Option<DropoutLayer>,
    pub(crate) input: Linear<A, K, D>,
    pub(crate) output: Linear<A, K, D>,
}

impl<A, K> FeedForwardNetwork<A, Ix2, K>
where
    K: ParamMode,
{
    pub fn new(d_model: usize, features: usize, dropout: Option<f64>) -> Self
    where
        A: Clone + Default,
    {
        let dropout = dropout.map(|p| DropoutLayer::new(p));
        let input = Linear::from_features(d_model, features);
        let output = Linear::from_features(features, d_model);
        Self {
            dropout,
            input,
            output,
        }
    }
}

impl<A, D, K> FeedForwardNetwork<A, D, K>
where
    D: Dimension,
{
    pub fn input(&self) -> &Linear<A, K, D> {
        &self.input
    }

    pub fn output(&self) -> &Linear<A, K, D> {
        &self.output
    }
}

#[cfg(feature = "rand")]
impl<A, D, K> FeedForwardNetwork<A, D, K>
where
    D: Dimension,
{
    pub fn dropout(&self) -> Option<&DropoutLayer> {
        self.dropout.as_ref()
    }
}

#[cfg(not(feature = "rand"))]
impl<A, D, K> FeedForwardNetwork<A, D, K>
where
    D: Dimension,
{
    pub fn dropout(&self) -> Option<&DropoutLayer> {
        None
    }
}

impl<A, B, D, E, K> Predict<Array<B, E>> for FeedForwardNetwork<A, D, K>
where
    B: Num + PartialOrd + ScalarOperand,
    D: RemoveAxis,
    E: Dimension,
    Linear<A, K, D>: Predict<Array<B, E>, Output = Array<B, E>>,
{
    type Output = Array<B, E>;

    fn predict(&self, input: &Array<B, E>) -> Result<Self::Output, PredictError> {
        let mut y = self.input().predict(input)?.relu();
        if let Some(dropout) = self.dropout() {
            y = dropout.forward(&y);
        }
        self.output().predict(&y)
    }
}
