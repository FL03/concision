/*
    Appellation: encoder <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::layer::EncoderLayer;

pub mod layer;

#[derive(Default)]
pub struct Encoder {}

impl Encoder {
    pub fn new() -> Self {
        Self {}
    }
}
