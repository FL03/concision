/*
   Appellation: nn <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{error::ModelError, models::Module};

pub mod error;
pub mod models;

pub(crate) mod prelude {
    pub use super::error::ModelError;
    pub use super::models::prelude::*;
}

#[cfg(test)]
mod tests {}
