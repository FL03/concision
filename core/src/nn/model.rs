/*
    Appellation: model <traits>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::module::*;

pub mod config;
pub mod module;

pub(crate) mod prelude {
    pub use super::config::*;
    pub use super::module::*;
    pub use super::Model;
}

use crate::traits::Forward;

pub trait Model: Module
where
    Self: Forward<Self::Data>,
{
    type Ctx;
    type Data;

    fn children(&self) -> Vec<ModuleDyn<Self::Ctx, Self::Params>>;

    fn context(&self) -> Self::Ctx;
}

