/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Parameters
//!
//! ## Overview
//!
pub use self::{group::*, param::*, shapes::*};

pub(crate) mod group;
pub(crate) mod param;
pub(crate) mod shapes;

use ndarray::linalg::Dot;
use ndarray::prelude::{Array, Dimension, Ix2};
use ndarray::IntoDimension;
use num::Float;

pub type BoxedParams<T = f64, D = Ix2> = Box<dyn Params<T, D>>;

pub trait Biased<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Array<T, D::Smaller>;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Array<T, D::Smaller>;
    /// Sets the bias of the layer.
    fn set_bias(&mut self, bias: Array<T, D::Smaller>);
}

pub trait Weighted<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    /// Returns an owned reference to the weights of the layer.
    fn weights(&self) -> &Array<T, D>;
    /// Returns a mutable reference to the weights of the layer.
    fn weights_mut(&mut self) -> &mut Array<T, D>;
    /// Sets the weights of the layer.
    fn set_weights(&mut self, weights: Array<T, D>);
}

pub trait Params<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    /// Returns an owned reference to the bias of the layer.
    fn bias(&self) -> &Array<T, D::Smaller>;
    /// Returns a mutable reference to the bias of the layer.
    fn bias_mut(&mut self) -> &mut Array<T, D::Smaller>;
    /// Returns an owned reference to the weights of the layer.
    fn weights(&self) -> &Array<T, D>;
    /// Returns a mutable reference to the weights of the layer.
    fn weights_mut(&mut self) -> &mut Array<T, D>;
    /// Sets the bias of the layer.
    fn set_bias(&mut self, bias: Array<T, D::Smaller>);
    /// Sets the weights of the layer.
    fn set_weights(&mut self, weights: Array<T, D>);
}

pub trait ParamsExt<T = f64, D = Ix2>: Biased<T, D> + Weighted<T, D>
where
    Array<T, D>: Dot<Array<T, D>, Output = Array<T, D>>,
    D: Dimension,
    T: Float,
{
    fn linear(&self, args: &Array<T, D>) -> Array<T, D> {
        args.dot(self.weights()) + self.bias()
    }
}

pub trait Parameterized<T = f64, D = Ix2>
where
    D: Dimension,
    T: Float,
{
    type Features: IntoDimension<Dim = D>;
    type Params;

    fn features(&self) -> &Self::Features;

    fn features_mut(&mut self) -> &mut Self::Features;

    fn params(&self) -> &Self::Params;

    fn params_mut(&mut self) -> &mut Self::Params;
}

pub trait ParameterizedExt<T = f64, D = Ix2>: Parameterized<T, D>
where
    D: Dimension,
    T: Float,
    <Self as Parameterized<T, D>>::Params: Params<T, D> + 'static,
{
    fn bias(&self) -> &Array<T, D::Smaller> {
        Params::bias(self.params())
    }

    fn bias_mut(&mut self) -> &mut Array<T, D::Smaller> {
        Params::bias_mut(self.params_mut())
    }

    fn weights(&self) -> &Array<T, D> {
        Params::weights(self.params())
    }

    fn weights_mut(&mut self) -> &mut Array<T, D> {
        Params::weights_mut(self.params_mut())
    }

    fn set_bias(&mut self, bias: Array<T, D::Smaller>) {
        Params::set_bias(self.params_mut(), bias)
    }

    fn set_weights(&mut self, weights: Array<T, D>) {
        Params::set_weights(self.params_mut(), weights)
    }
}

impl<T, D, P> ParameterizedExt<T, D> for P
where
    D: Dimension,
    P: Parameterized<T, D>,
    T: Float,
    <P as Parameterized<T, D>>::Params: Params<T, D> + 'static,
{
}

// impl<S, T, D, P> Params<T, D> for S
// where
//     S: Parameterized<T, D, Params = P>,
//     D: Dimension,
//     P: Biased<T, D>,
//     T: Float,
//     <D as Dimension>::Smaller: Dimension,
// {
//     fn bias(&self) -> &Array<T, D::Smaller> {
//         self.params().bias()
//     }

//     fn bias_mut(&mut self) -> &mut Array<T, D::Smaller> {
//         self.params_mut().bias_mut()
//     }

//     fn weights(&self) -> &Array<T, D> {
//         self.params().weights()
//     }

//     fn weights_mut(&mut self) -> &mut Array<T, D> {
//         self.params_mut().weights_mut()
//     }

//     fn set_bias(&mut self, bias: Array<T, D::Smaller>) {
//         self.params_mut().set_bias(bias)
//     }

//     fn set_weights(&mut self, weights: Array<T, D>) {
//         self.params_mut().set_weights(weights)
//     }
// }

impl<T, D, P> Params<T, D> for P
where
    D: Dimension,
    T: Float,
    Self: Biased<T, D> + Weighted<T, D> + Sized,
{
    fn bias(&self) -> &Array<T, D::Smaller> {
        Biased::bias(self)
    }

    fn bias_mut(&mut self) -> &mut Array<T, D::Smaller> {
        Biased::bias_mut(self)
    }

    fn weights(&self) -> &Array<T, D> {
        Weighted::weights(self)
    }

    fn weights_mut(&mut self) -> &mut Array<T, D> {
        Weighted::weights_mut(self)
    }

    fn set_bias(&mut self, bias: Array<T, D::Smaller>) {
        Biased::set_bias(self, bias)
    }

    fn set_weights(&mut self, weights: Array<T, D>) {
        Weighted::set_weights(self, weights)
    }
}

// impl<T, D, P> Biased<T, D> for P
// where
//     D: Dimension,
//     P: Parameterized<T, D>,
//     T: Float,
//     <D as Dimension>::Smaller: Dimension,
//     <P as Parameterized<T, D>>::Params: 'static,
// {
//     fn bias(&self) -> &Array<T, D::Smaller> {
//         self.params().bias()
//     }

//     fn bias_mut(&mut self) -> &mut Array<T, D::Smaller> {
//         self.params_mut().bias_mut()
//     }
// }

#[cfg(test)]
mod tests {}
