/*
   Appellation: errors <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{error::*, kinds::*};

pub(crate) mod error;
pub(crate) mod kinds;

pub trait KindOf {
    type Kind;

    fn kind(&self) -> Self::Kind;
}

#[cfg(test)]
mod tests {}
