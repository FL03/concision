/*
    Appellation: predict <module>
    Contrib: @FL03
*/

/// This trait defines the forward pass of the network

pub trait Forward<Rhs> {
    type Output;

    fn forward(&self, input: &Rhs) -> crate::CncResult<Self::Output>;
}

/// This trait defines the prediction of the network
pub trait Predict<Rhs> {
    type Confidence;
    type Output;

    fn predict(&self, input: &Rhs) -> crate::CncResult<Self::Output>;

    fn predict_with_confidence(
        &self,
        input: &Rhs,
    ) -> crate::CncResult<(Self::Output, Self::Confidence)>;
}
