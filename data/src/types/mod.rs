/*
   Appellation: types <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod direction;

pub(crate) mod prelude {
   pub use super::direction::Direction;
}

#[cfg(test)]
mod tests {}
