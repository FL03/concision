/*
   Appellation: nn <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{error::ModelError, model::prelude::*};

pub mod error;
pub mod model;

pub(crate) mod prelude {
    pub use super::error::ModelError;
    pub use super::model::prelude::*;
}

#[cfg(test)]
mod tests {}
