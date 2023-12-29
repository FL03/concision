/*
   Appellation: errors <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{error::*, utils::*};

pub(crate) mod error;

pub(crate) mod utils {

    pub fn random_err() -> String {
        String::new()
    }
}

#[cfg(test)]
mod tests {}
