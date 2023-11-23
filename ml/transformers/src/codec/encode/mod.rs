/*
   Appellation: encode <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Encode
pub use self::{encoder::*, params::*, stack::*, utils::*};

pub(crate) mod encoder;
pub(crate) mod params;
pub(crate) mod stack;

pub trait Encode {}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::Mask;
    use ndarray::Array2;

    #[test]
    fn test_encoder() {
        let (heads, seq, model) = (8, 10, 512);
        let _data = Array2::<f64>::zeros((seq, model));
        let _mask = Mask::<f64>::masked(seq);
        let params = EncoderParams::new(heads, model);
        let encoder = Encoder::new(params);

        assert_eq!(encoder.params().heads(), heads);
    }
}
