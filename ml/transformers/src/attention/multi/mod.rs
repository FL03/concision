/*
   Appellation: multi <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{attention::*, utils::*};

pub(crate) mod attention;

pub trait Split {
    fn split(&self) -> Vec<Self>
    where
        Self: Sized;
}

pub(crate) mod utils {}
