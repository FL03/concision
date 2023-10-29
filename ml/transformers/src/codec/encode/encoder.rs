/*
   Appellation: encoder <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::EncoderParams;
use crate::attention::multi::MultiHeadAttention;
use crate::ffn::FFN;


pub struct Encoder {
   attention: MultiHeadAttention,
   network: FFN,
   params: EncoderParams,
}

impl Encoder {
   pub fn new(params: EncoderParams) -> Self {
      let attention = MultiHeadAttention::new(params.heads, params.model);
      let network = FFN::new(params.model, None);
      Self { attention, network, params,  }
   }
}
