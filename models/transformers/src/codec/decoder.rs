/*
    Appellation: decoder <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::layer::DecoderLayer;

pub mod layer;

#[derive(Default)]
pub struct Decoder {}

impl Decoder {
    pub fn new() -> Self {
        Self {}
    }
}
