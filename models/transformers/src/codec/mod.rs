/*
    Appellation: codec <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{decoder::Decoder, encoder::Encoder, model::*};

pub(crate) mod model;

pub mod decoder;
pub mod encoder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_builder() {
        let ctx = Context::new()
            .with_src("src".to_string())
            .with_tgt("tgt".to_string());
        let codec = Codec::new().ctx(ctx).build();
        assert_eq!(codec.context().src, "src");
        assert_eq!(codec.context().tgt, "tgt");
    }
}
