/*
    Appellation: loss <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#[doc(inline)]
pub use self::{entropy::*, reg::*};

pub mod entropy;
pub mod reg;

pub(crate) mod prelude {
    pub use super::Loss;
    pub use super::reg::*;
}

pub trait Loss<A, B = A> {
    type Output;

    fn loss(&self, a: &A, cmp: &B) -> Self::Output;
}
