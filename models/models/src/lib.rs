/*
    Appellation: concision-models <library>
    Contrib: @FL03
*/


#[cfg(feature = "simple")]
pub mod simple;


pub mod prelude [
    #[cfg(feature = "simple")]
    pub use simple::prelude::*;
]