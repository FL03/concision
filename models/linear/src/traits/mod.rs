/*
    Appellation: traits <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::wnb::*;

pub mod wnb;

pub(crate) mod prelude {
    pub use super::wnb::*;
}
