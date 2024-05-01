/*
   Appellation: models <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{model::*, traits::prelude::*};

pub(crate) mod model;

pub(crate) mod traits {
    mod model;
    mod modules;

    pub(crate) mod prelude {
        pub use super::model::*;
        pub use super::modules::*;
    }
}

pub(crate) mod prelude {
    pub use super::model::*;
    pub use super::traits::prelude::*;
}

#[cfg(test)]
mod tests {}
