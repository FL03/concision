/*
   Appellation: stack <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Encoder, EncoderParams};

pub struct EncoderStack {
    layers: usize,
    params: EncoderParams,
    stack: Vec<Encoder>,
}

impl EncoderStack {
    pub fn new(layers: usize, params: EncoderParams) -> Self {
        let stack = Vec::with_capacity(layers);

        Self {
            layers,
            params,
            stack,
        }
    }

    pub fn setup(&mut self) {
        for _ in 0..self.layers {
            self.stack.push(Encoder::new(self.params));
        }
    }
}
