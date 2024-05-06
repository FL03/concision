/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::LinearParamsBase;
use crate::{Biased, Weighted};
use concision::prelude::{Predict, PredictError};
use core::ops::Add;
use nd::linalg::Dot;
use nd::*;

impl<A, S, D> Biased for LinearParamsBase<S, D>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    type Bias = ArrayBase<S, D::Smaller>;

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

impl<A, S, D> Weighted for LinearParamsBase<S, D>
where
    D: RemoveAxis,
    S: RawData<Elem = A>,
{
    type Weight = ArrayBase<S, D>;

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

impl<A, B, T, S, D> Predict<A> for LinearParamsBase<S, D>
where
    A: Dot<Array<T, D>, Output = B>,
    B: for<'a> Add<&'a ArrayBase<S, D::Smaller>, Output = B>,
    D: RemoveAxis,
    S: Data<Elem = T>,
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

impl<'a, A, B, T, S, D> Predict<A> for &'a LinearParamsBase<S, D>
where
    A: Dot<Array<T, D>, Output = B>,
    B: Add<&'a ArrayBase<S, D::Smaller>, Output = B>,
    D: RemoveAxis,
    S: Data<Elem = T>,
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

impl<A, S, D> Clone for LinearParamsBase<S, D>
where
    A: Clone,
    D: RemoveAxis,
    S: RawDataClone<Elem = A>,
{
    fn clone(&self) -> Self {
        Self {
            weights: self.weights.clone(),
            bias: self.bias.clone(),
        }
    }
}
