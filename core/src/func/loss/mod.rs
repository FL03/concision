/*
    Appellation: loss <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::reg::prelude::*;
pub use self::{entropy::*, utils::*};

pub(crate) mod utils;

pub mod entropy;
pub mod reg;

pub(crate) mod prelude {
    pub use super::Loss;
    pub use super::reg::prelude::*;
    pub use super::utils::*;
}

pub trait Loss<A, B = A> {
    type Output;

    fn loss(&self, a: &A, cmp: &B) -> Self::Output;
}
