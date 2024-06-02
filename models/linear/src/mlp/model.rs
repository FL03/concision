/*
    Appellation: model <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use concision::{Predict, PredictError};
use core::marker::PhantomData;

// #92: Define the Multi-Layer Perceptron (MLP) model
/// A multi-layer perceptron (MLP) model.
pub struct Mlp<A, I, H, O>
where
    I: Predict<A>,
    H: Predict<I::Output, Output = I::Output>,
    O: Predict<H::Output>,
{
    input: I,
    hidden: Vec<H>,
    output: O,
    _dtype: PhantomData<A>,
}

impl<A, I, H, O> Predict<A> for Mlp<A, I, H, O>
where
    I: Predict<A>,
    H: Predict<I::Output, Output = I::Output>,
    O: Predict<H::Output>,
{
    type Output = O::Output;

    fn predict(&self, input: &A) -> Result<Self::Output, PredictError> {
        let mut hidden = self.input.predict(input)?;
        for layer in &self.hidden {
            hidden = layer.predict(&hidden)?;
        }
        self.output.predict(&hidden)
    }
}
