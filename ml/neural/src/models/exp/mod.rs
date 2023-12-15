/*
    Appellation: exp <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Experimental Models
pub use self::{modules::*, store::*, utils::*};

pub(crate) mod modules;
pub(crate) mod store;

use crate::prelude::Predict;
use num::Float;

pub trait Model<T = f64>: Predict<T> where T: Float {
    type Config;
    
    fn name(&self) -> &str;

    fn modules(&self) -> &Vec<Box<dyn Module<T, Output = Self::Output>>>;

    fn modules_mut(&mut self) -> &mut Vec<Box<dyn Module<T, Output = Self::Output>>>;

    fn register(&mut self, module: Box<dyn Module<T, Output = Self::Output>>) -> &mut Self {
        self.modules_mut().push(module);
        self
    }
}

pub(crate) mod utils {}

#[cfg(test)]
mod tests {}
