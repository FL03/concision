/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::LinearParams;
use crate::{Biased, Weighted};
use concision::prelude::{Predict, PredictError};
use core::ops::Add;
use nd::linalg::Dot;
use nd::*;
use num::Float;

impl<T, D> Biased for LinearParams<T, D>
where
    D: RemoveAxis,
    T: Float,
{
    type Bias = Array<T, D::Smaller>;

    fn bias(&self) -> &Self::Bias {
        self.bias.as_ref().unwrap()
    }

    fn bias_mut(&mut self) -> &mut Self::Bias {
        self.bias.as_mut().unwrap()
    }

    fn set_bias(&mut self, bias: Self::Bias) {
        self.bias = Some(bias);
    }
}

impl<T, D> Weighted for LinearParams<T, D>
where
    D: Dimension,
{
    type Weight = Array<T, D>;

    fn weights(&self) -> &Self::Weight {
        &self.weights
    }

    fn weights_mut(&mut self) -> &mut Self::Weight {
        &mut self.weights
    }

    fn set_weights(&mut self, weights: Self::Weight) {
        self.weights = weights;
    }
}

impl<A, B, T, D> Predict<A> for LinearParams<T, D>
where
    A: Dot<Array<T, D>, Output = B>,
    B: for<'a> Add<&'a Array<T, D::Smaller>, Output = B>,
    D: RemoveAxis,
    T: NdFloat,
{
    type Output = B;

    fn predict(&self, input: &A) -> Result<Self::Output, PredictError> {
        let wt = self.weights().t().to_owned();
        let res = input.dot(&wt);
        if let Some(bias) = self.bias() {
            return Ok(res + bias);
        }
        Ok(res)
    }
}

impl<'a, A, B, T, D> Predict<A> for &'a LinearParams<T, D>
where
    A: Dot<Array<T, D>, Output = B>,
    B: Add<&'a Array<T, D::Smaller>, Output = B>,
    D: RemoveAxis,
    T: NdFloat,
{
    type Output = B;

    fn predict(&self, input: &A) -> Result<Self::Output, PredictError> {
        let wt = self.weights().t().to_owned();
        let res = input.dot(&wt);
        if let Some(bias) = self.bias() {
            return Ok(res + bias);
        }
        Ok(res)
    }
}
