/*
    Appellation: concision-models <library>
    Contrib: @FL03
*/

#[cfg(feature = "simple")]
pub mod simple;
#[cfg(feature = "transformer")]
pub use transformer;

pub mod prelude {
    #[cfg(feature = "simple")]
    pub use crate::simple::SimpleModel;
}
