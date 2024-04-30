/*
   Appellation: models <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::model::*;

pub(crate) mod model;

pub(crate) mod prelude {
    pub use super::model::*;
}

#[cfg(test)]
mod tests {}
