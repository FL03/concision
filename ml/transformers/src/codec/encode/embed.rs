/*
   Appellation: embed <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Embed {}

#[derive(Clone)]
pub struct Embedding {
   pub(crate) embedding: Vec<f32>,
   pub(crate) position: Vec<f32>,
}


