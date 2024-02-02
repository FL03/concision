/*
   Appellation: layout <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::shape::Shape;

pub struct Layout {
    shape: Shape,
    stride: Vec<usize>,
}

impl Layout {
    pub fn new(shape: Shape, stride: Vec<usize>) -> Self {
        Self { shape, stride }
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn stride(&self) -> &[usize] {
        &self.stride
    }
}
