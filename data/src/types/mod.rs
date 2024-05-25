/*
    Appellation: types <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::kernel::Kernel;

pub mod kernel;

pub(crate) mod prelude {
    pub use super::kernel::Kernel;
}
