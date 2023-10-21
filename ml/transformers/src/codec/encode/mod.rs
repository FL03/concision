/*
   Appellation: encode <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Encode
pub use self::{embed::*, encoder::*, utils::*};

pub(crate) mod embed;
pub(crate) mod encoder;

pub trait Encode {}

pub(crate) mod utils {
    use ndarray::Array2;

    pub fn get_position_encoding(seq_len: usize, d: usize, n: f64) -> Array2<f64> {
        let mut p = Array2::zeros((seq_len, d));
        for k in 0..seq_len {
            for i in 0..d / 2 {
                let denominator = f64::powf(n, 2.0 * i as f64 / d as f64);
                p[[k, 2 * i]] = (k as f64 / denominator).sin();
                p[[k, 2 * i + 1]] = (k as f64 / denominator).cos();
            }
        }
        p
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_encoder() {}

    #[test]
    fn test_positional_encoding() {
        let p = get_position_encoding(4, 4, 1000.);
        assert_eq!(p.row(0), array![0.0, 1.0, 0.0, 1.0]);
    }
}
