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

impl Embedding {
    pub fn new(embedding: Vec<f32>, position: Vec<f32>) -> Self {
        Self {
            embedding,
            position,
        }
    }

    pub fn embedding(&self) -> &Vec<f32> {
        &self.embedding
    }

    pub fn position(&self) -> &Vec<f32> {
        &self.position
    }
}
