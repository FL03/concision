/*
   Appellation: nn <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{activate::Activator, error::ModelError, models::Module, traits::prelude::*};

pub mod activate;
pub mod error;
pub mod models;

pub mod traits {
    pub use self::prelude::*;

    pub mod predict;
    pub mod train;
    pub(crate) mod prelude {
        pub use super::predict::*;
        pub use super::train::*;
    }
}
pub(crate) mod prelude {
    pub use super::activate::Activator;
    pub use super::error::ModelError;
    pub use super::models::prelude::*;
    pub use super::traits::prelude::*;
}

#[cfg(test)]
mod tests {}
