/*
   Appellation: layout <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::shape::Shape;

pub struct Layout {
    shape: Shape,
}

impl Layout {
    pub fn new(shape: Shape) -> Self {
        Self { shape }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}
