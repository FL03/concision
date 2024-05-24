/*
    Appellation: sublayer <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(feature = "rand")]
use concision::nn::DropoutLayer;
use concision::Forward;
use linear::{Biased, LayerNorm, ParamMode, Unbiased};
use nd::prelude::*;
use nd::{DataOwned, RemoveAxis, ScalarOperand};
use num::traits::{Float, FromPrimitive};

/// A residual connection followed by a [layer norm](LayerNorm)
/// [Transformer](crate::Transformer)
pub struct Sublayer<A = f64, K = Biased, D = Ix2>
where
    D: Dimension,
{
    pub(crate) dropout: DropoutLayer,
    pub(crate) norm: LayerNorm<A, K, D>,
}

impl<A, K, D> Sublayer<A, K, D>
where
    D: RemoveAxis,
{
    pub fn new<Sh>(shape: Sh, dropout: f64) -> Self
    where
        A: Default,
        K: ParamMode,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self {
            dropout: DropoutLayer::new(dropout),
            norm: LayerNorm::new(shape),
        }
    }

    pub fn dropout(&self) -> &DropoutLayer {
        &self.dropout
    }

    pub fn norm(&self) -> &LayerNorm<A, K, D> {
        &self.norm
    }
}

impl<A, S, D> Forward<ArrayBase<S, D>> for Sublayer<A, Biased, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: RemoveAxis,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn forward(&self, input: &ArrayBase<S, D>) -> Self::Output {
        let normal = self.norm().forward(input);
        input + self.dropout().forward(&normal)
    }
}

impl<A, S, D> Forward<ArrayBase<S, D>> for Sublayer<A, Unbiased, D>
where
    A: Float + FromPrimitive + ScalarOperand,
    D: RemoveAxis,
    S: DataOwned<Elem = A>,
{
    type Output = Array<A, D>;

    fn forward(&self, input: &ArrayBase<S, D>) -> Self::Output {
        let normal = self.norm().forward(input);
        input + self.dropout().forward(&normal)
    }
}
