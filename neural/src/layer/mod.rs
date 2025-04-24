/*
    Appellation: layers <module>
    Contrib: @FL03
*/
//!
#[doc(inline)]
pub use self::layer::LayerBase;

pub(crate) mod layer;

pub(crate) mod prelude {
    pub use super::layer::*;
}

use cnc::ParamsBase;

pub trait BinaryAction<A, B = A> {
    type Output;

    fn activate(lhs: A, rhs: B) -> Self::Output;
}

pub trait Activate<Rhs = Self> {
    type Output;

    fn activate(&self, rhs: Rhs) -> Self::Output;
}

impl<X, Y, F> Activate<X> for F
where
    F: Fn(X) -> Y,
{
    type Output = Y;

    fn activate(&self, rhs: X) -> Self::Output {
        self(rhs)
    }
}

use ndarray::{Dimension, Ix2, RawData};
pub trait Layer<S, D = Ix2>
where
    D: Dimension,
    S: RawData,
{
    /// returns an immutable reference to the parameters of the layer
    fn params(&self) -> &ParamsBase<S, D>;
    /// returns a mutable reference to the parameters of the layer
    fn params_mut(&mut self) -> &mut ParamsBase<S, D>;

    type Rho<U, V>: Activate<U, Output = V>;

    fn rho<A, B>(&self) -> &Self::Rho<A, B>;
    ///
    fn forward<X, Y>(&self, input: &X) -> cnc::CncResult<Y>
    where
        S::Elem: Clone,
        S: ndarray::Data,
        ParamsBase<S, D>: cnc::Forward<X, Output = Y>,
    {
        self.params().forward(input).map(|y| self.rho().activate(y))
    }
}
