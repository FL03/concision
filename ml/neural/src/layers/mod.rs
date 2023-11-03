/*
    Appellation: layers <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Layers
pub use self::{features::*, kinds::*, layer::*, sublayer::*, utils::*};

pub(crate) mod features;
pub(crate) mod kinds;
pub(crate) mod layer;
pub(crate) mod sublayer;

pub mod linear;

use crate::neurons::activate::Activate;
use ndarray::prelude::{Array1, Array2};
use num::Float;

pub trait L<T: Float> {
    //
    fn process(&self, args: &Array2<T>, rho: impl Activate<T>) -> Array2<T>
    where
        T: 'static,
    {
        let z = args.dot(self.weights()) + self.bias();
        z.mapv(|x| rho.activate(x))
    }

    fn bias(&self) -> &Array1<T>;

    fn weights(&self) -> &Array2<T>;
}

pub trait Linear<T: Float> {
    fn linear(&self, data: &Array2<T>) -> Array2<T>
    where
        T: 'static;
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    // use super::*;

    #[test]
    fn test_layer() {}
}
