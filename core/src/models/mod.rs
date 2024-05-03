/*
   Appellation: models <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{activate::Activator, error::ModelError};

pub mod activate;
pub mod error;

pub(crate) mod prelude {
    pub use super::activate::Activator;
    pub use super::error::ModelError;
}

#[cfg(test)]
mod tests {}
