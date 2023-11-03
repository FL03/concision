/*
   Appellation: positional <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

use ndarray::Array2;

pub fn get_position_encoding(seq_len: usize, d: usize, n: f64) -> Array2<f64> {
    let denom = |i: usize| f64::powf(n, 2.0 * (i as f64) / d as f64);
    let mut p = Array2::zeros((seq_len, d));
    for k in 0..seq_len {
        for i in 0..d / 2 {
            p[[k, 2 * i]] = (k as f64 / denom(i)).sin();
            p[[k, 2 * i + 1]] = (k as f64 / denom(i)).cos();
        }
    }
    p
}

pub struct PositionalEncoder {
    model: usize,
    sequence: usize,
    samples: usize,
}

impl PositionalEncoder {
    pub fn new(model: usize, sequence: usize, samples: usize) -> Self {
        Self {
            model,
            sequence,
            samples,
        }
    }

    pub fn encode(&self, data: &Array2<f64>) -> Array2<f64> {
        let x = data * (self.model as f64).sqrt();
        x + self.positional()
    }

    pub fn positional(&self) -> Array2<f64> {
        get_position_encoding(self.sequence, self.model, self.samples as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_positional_encoding() {
        let p = get_position_encoding(4, 4, 10000.);
        assert_eq!(p.row(0), array![0.0, 1.0, 0.0, 1.0]);
    }
}
