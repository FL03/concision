/*
    Appellation: config <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::cmp::LinearShape;

#[derive(Clone, Debug)]
pub struct LinearConfig {
    pub biased: bool,
    pub name: String,
    pub shape: LinearShape,
}

impl LinearConfig {
    pub fn new(shape: LinearShape) -> Self {
        Self {
            biased: false,
            name: String::new(),
            shape,
        }
    }

    pub fn is_biased(&self) -> bool {
        self.biased
    }

    pub fn biased(self) -> Self {
        Self {
            biased: true,
            ..self
        }
    }

    pub fn unbiased(self) -> Self {
        Self {
            biased: false,
            ..self
        }
    }

    pub fn with_name(mut self, name: impl ToString) -> Self {
        self.name = name.to_string();
        self
    }

    pub fn with_shape(mut self, shape: LinearShape) -> Self {
        self.shape = shape;
        self
    }
}
