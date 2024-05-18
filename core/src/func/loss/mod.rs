/*
    Appellation: loss <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub mod entropy;
pub mod reg;

pub(crate) mod prelude {
    pub use super::reg::prelude::*;
    pub use super::Loss;
}

pub trait Loss<T> {
    type Output;

    fn loss(&self, cmp: &T) -> Self::Output;
}
