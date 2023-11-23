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

use crate::func::activate::{Activate, ActivateDyn};
use crate::prelude::Forward;
use ndarray::prelude::{Array2, Ix2};
use num::Float;

pub type LayerDyn<T = f64> = Layer<T, ActivateDyn<T, Ix2>>;

pub trait L<T: Float>: Forward<Array2<T>> {
    fn features(&self) -> &LayerShape;

    fn features_mut(&mut self) -> &mut LayerShape;

    fn params(&self) -> &LayerParams<T>;

    fn params_mut(&mut self) -> &mut LayerParams<T>;
}

pub trait LayerExt<T = f64>: L<T>
where
    T: Float,
{
    type Rho: Activate<T, Ix2>;
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
        let features = LayerShape::new(inputs, outputs);

        let args = linarr::<f64, Ix2>((samples, inputs)).unwrap();

        let layer = Layer::<f64, Softmax>::from(features).init(true);

        let pred = layer.forward(&args);

        assert_eq!(pred.dim(), (samples, outputs));
    }
}
