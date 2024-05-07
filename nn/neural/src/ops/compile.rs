/*
   Appellation: compile <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::core::prelude::BoxResult;
use crate::func::loss::Loss;

pub trait Compile<T> {
    type Opt;

    fn compile(&mut self, loss: impl Loss<T>, optimizer: Self::Opt) -> BoxResult<()>;
}
