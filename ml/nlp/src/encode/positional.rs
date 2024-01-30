/*
   Appellation: positional <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use ndarray::prelude::Array2;
use ndarray::ScalarOperand;
use num::Float;
use serde::{Deserialize, Serialize};

pub fn create_positional<T: Float>(
    model: usize,
    seq_len: usize,
    samples: Option<usize>,
) -> Array2<T> {
    let n = T::from(samples.unwrap_or(10000)).unwrap();
    let d = T::from(model).unwrap();
    let denom = |pos: T, x: T| pos / T::powf(n, (T::from(2).unwrap() * x) / d);
    let mut p = Array2::zeros((seq_len, model));
    for i in 0..seq_len {
        for j in 0..model / 2 {
            let u = T::from(i).unwrap();
            let v = T::from(j).unwrap();
            p[[i, 2 * j]] = denom(u, v).sin();
            p[[i, 2 * j + 1]] = denom(u, v + T::one()).cos();
        }
    }
    p
}

#[derive(Clone, Debug, Default, Deserialize, Eq, Hash, PartialEq, Serialize)]
pub struct PositionalEncoder<T = f64> {
    params: PositionalEncoderParams,
    pe: Array2<T>,
}

impl<T> PositionalEncoder<T>
where
    T: Float,
{
    pub fn new(model: usize, sequence: usize, samples: usize) -> Self {
        Self {
            params: PositionalEncoderParams::new(model, sequence, samples),
            pe: Array2::zeros((sequence, model)),
        }
    }

    pub fn init(mut self) -> Self {
        self.pe = create_positional::<T>(
            self.params.model(),
            self.params.sequence(),
            Some(self.params.samples()),
        );
        self
    }

    pub fn params(&self) -> PositionalEncoderParams {
        self.params
    }

    pub fn positional(&self) -> &Array2<T> {
        &self.pe
    }
}

impl<T> PositionalEncoder<T>
where
    T: Float + ScalarOperand,
{
    pub fn encode(&self, data: &Array2<T>) -> Array2<T> {
        let x = data * T::from(self.params().model()).unwrap().sqrt();
        x + self.positional()
    }
}

pub trait IntoParams {
    type Params;

    fn into_params(self) -> Self::Params;
}

#[derive(
    Clone, Copy, Debug, Default, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize,
)]
pub struct PositionalEncoderParams {
    pub model: usize,
    pub sequence: usize,
    pub samples: usize,
}

impl PositionalEncoderParams {
    pub fn new(model: usize, sequence: usize, samples: usize) -> Self {
        Self {
            model,
            sequence,
            samples,
        }
    }

    pub fn std(sequence: usize) -> Self {
        Self::new(512, sequence, 10000)
    }

    pub fn model(&self) -> usize {
        self.model
    }

    pub fn sequence(&self) -> usize {
        self.sequence
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn set_model(&mut self, model: usize) {
        self.model = model;
    }

    pub fn set_sequence(&mut self, sequence: usize) {
        self.sequence = sequence;
    }

    pub fn set_samples(&mut self, samples: usize) {
        self.samples = samples;
    }

    pub fn with_model(mut self, model: usize) -> Self {
        self.model = model;
        self
    }

    pub fn with_sequence(mut self, sequence: usize) -> Self {
        self.sequence = sequence;
        self
    }

    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::prelude::RoundTo;
    use ndarray::prelude::{array, Array};

    #[test]
    fn test_positional_encoding() {
        let data = Array::linspace(1., 4., 4).into_shape((1, 4)).unwrap();
        let encoder = PositionalEncoder::new(4, 4, 10000).init();

        let pe = encoder.positional();
        assert_eq!(pe.dim(), (4, 4));
        assert_eq!(pe.row(0), array![0.0, 1.0, 0.0, 1.0]);

        let encoded = encoder.encode(&data);
        let rounded = encoded.mapv(|x| x.round_to(4));

        assert_eq!(rounded[[0, 0]], 2.0);
        assert_eq!(rounded[[1, 0]], 2.8415);
    }
}
