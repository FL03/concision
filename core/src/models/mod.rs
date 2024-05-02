/*
   Appellation: models <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{error::ModelError, model::*, traits::prelude::*};

pub(crate) mod model;

pub mod error;

pub(crate) mod traits {
    mod model;
    mod modules;

    pub(crate) mod prelude {
        pub use super::model::*;
        pub use super::modules::*;
    }
}

pub(crate) mod prelude {
    pub use super::error::ModelError;
    pub use super::model::*;
    pub use super::traits::prelude::*;
}

#[cfg(test)]
mod tests {}
