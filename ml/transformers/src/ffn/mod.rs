/*
   Appellation: decode <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Decode
pub use self::{network::*, params::*, utils::*};

pub(crate) mod network;
pub(crate) mod params;

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::linarr;
    use crate::neural::prelude::Forward;
    use ndarray::prelude::Ix2;

    #[test]
    fn test_ffn() {
        let samples = 20;
        let (model, network) = (5, 15);

        // sample data
        let x = linarr::<f64, Ix2>((samples, model)).unwrap();
        let _y = linarr::<f64, Ix2>((samples, model)).unwrap();

        let ffn = FFN::new(model, Some(network));
        // assert!(network.validate_dims());

        let pred = ffn.forward(&x);
        assert_eq!(&pred.dim(), &(samples, model));
    }
}
