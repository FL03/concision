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
    use ndarray::Array2;

    #[test]
    fn test_encoder() {
        let (heads, model) = (8, 512);
        let _data = Array2::<f64>::zeros((512, 512));
        let params = EncoderParams::new(heads, model);
        let encoder = Encoder::new(params);
        assert_eq!(encoder.params().heads(), heads);
    }
}
