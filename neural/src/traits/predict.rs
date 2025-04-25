/*
    Appellation: predict <module>
    Contrib: @FL03
*/

/// This trait defines the prediction of the network
pub trait Predict<Rhs> {
    type Confidence;
    type Output;

    fn predict(&self, input: &Rhs) -> crate::NeuralResult<Self::Output>;

    fn predict_with_confidence(
        &self,
        input: &Rhs,
    ) -> crate::NeuralResult<(Self::Output, Self::Confidence)>;
}
