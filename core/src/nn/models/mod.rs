/*
    Appellation: models <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{model::*, module::*};

pub mod model;
pub mod module;

pub(crate) mod prelude {
    pub use super::model::*;
    pub use super::module::*;
}
