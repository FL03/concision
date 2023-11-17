/*
    Appellation: layers <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Layers
pub use self::{features::*, kinds::*, layer::*, params::*, sublayer::*, utils::*};

pub(crate) mod features;
pub(crate) mod kinds;
pub(crate) mod layer;
pub(crate) mod params;
pub(crate) mod sublayer;

pub mod linear;

use crate::func::activate::Activate;
use ndarray::prelude::{Array1, Array2};
use num::Float;

pub trait L<T: Float> {
    fn forward_slice(&self, args: &Array2<T>, rho: impl Activate<T>) -> Array2<T>
    where
        T: 'static,
    {
        let z = args.dot(self.weights()) + self.bias();
        z.mapv(|x| rho.activate(x))
    }
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

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::linarr;
    use crate::func::activate::Softmax;
    use crate::prelude::Forward;
    use ndarray::prelude::Ix2;

    #[test]
    fn test_layer() {
        let (samples, inputs, outputs) = (20, 5, 3);
        let features = Features::new(inputs, outputs);

        let args = linarr::<f64, Ix2>((samples, inputs)).unwrap();

        let layer = Layer::<f64, Softmax>::from(features).init(true);

        let pred = layer.forward(&args);

        assert_eq!(pred.dim(), (samples, outputs));
    }
}
