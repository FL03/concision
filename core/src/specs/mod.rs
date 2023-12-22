/*
   Appellation: specs <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{arrays::*, base::*, math::*};

pub(crate) mod arrays;
pub(crate) mod base;
pub(crate) mod math;

#[cfg(test)]
mod tests {}
