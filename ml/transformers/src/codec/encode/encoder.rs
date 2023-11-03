/*
   Appellation: encoder <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::EncoderParams;
use crate::attention::multi::MultiHeadAttention;
use crate::ffn::FFN;
use crate::neural::prelude::{Forward, LayerNorm, Mask};
use ndarray::prelude::Array2;

pub struct Encoder {
    attention: MultiHeadAttention,
    network: FFN,
    norm_attention: LayerNorm,
    norm_network: LayerNorm,
    params: EncoderParams,
}

impl Encoder {
    pub fn new(params: EncoderParams) -> Self {
        let attention = MultiHeadAttention::new(params.heads, params.model);
        let network = FFN::new(params.model, None);
        Self {
            attention,
            network,
            norm_attention: LayerNorm::new(params.model),
            norm_network: LayerNorm::new(params.model),
            params,
        }
    }

    fn _forward(&self, data: &Array2<f64>, mask: &Mask<f64>) -> anyhow::Result<Array2<f64>> {
        let attention = data + self.attention.attention(data, mask)?;
        let norm = self.norm_attention.forward(&attention);
        let network = data + self.network.forward(&norm);
        let norm = self.norm_network.forward(&network);
        Ok(norm)
    }

    pub fn forward(&mut self, data: &Array2<f64>, mask: &Mask<f64>) -> anyhow::Result<Array2<f64>> {
        let norm = self.norm_attention.forward(data);
        let attention = data + self.attention.attention(&norm, mask)?;
        let norm = self.norm_network.forward(&attention);
        let network = data + self.network.forward(&norm);
        Ok(network)
    }

    pub fn params(&self) -> EncoderParams {
        self.params
    }
}
