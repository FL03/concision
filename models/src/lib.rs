/*
    Appellation: concision-models <library>
    Contrib: @FL03
*/


#[cfg(feature = "simple")]
pub use simple as simple;


pub mod prelude {
    #[cfg(feature = "simple")]
    pub use simple::SimpleModel;
}