/*
   Appellation: storage <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::shape::Shape;

pub struct Storage<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T> Storage<T> {
    pub fn new(data: Vec<T>, shape: Shape) -> Self {
        Self { data, shape }
    }

    pub fn data(&self) -> &Vec<T> {
        &self.data
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }
}
