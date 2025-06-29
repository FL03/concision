/*
    Appellation: seal <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! The public parts of this private module are used to create traits
//! that cannot be implemented outside of our own crate.  This way we
//! can feel free to extend those traits without worrying about it
//! being a breaking change for other implementations.
//!
//! ## Usage
//!
//! To define a private trait, you can use the [`private!`] macro, which will define a hidden
//! method `__private__` that can only be implemented within the crate.
//! ```

/// If this type is pub but not publicly reachable, third parties
/// can't name it and can't implement traits using it.
#[allow(dead_code)]
pub struct Seal;
/// the [`private!`] macro is used to seal a particular trait, defining a hidden method that
/// may only be implemented within the bounds of the crate.
#[allow(unused_macros)]
macro_rules! private {
    () => {
        /// This trait is private to implement; this method exists to make it
        /// impossible to implement outside the crate.
        #[doc(hidden)]
        fn __private__(&self) -> $crate::macros::seal::Seal;
    };
}
/// the [`seal!`] macro is used to implement a private method on a type, which is used to seal
/// the type so that it cannot be implemented outside of the crate.
#[allow(unused_macros)]
macro_rules! seal {
    () => {
        fn __private__(&self) -> $crate::macros::seal::Seal {
            $crate::macros::seal::Seal
        }
    };
}
/// this macros is used to implement a trait for a type, sealing it so that
/// it cannot be implemented outside of the crate. This is most usefuly for creating other
/// macros that can be used to implement some raw, sealed trait on the given _types_.
#[allow(unused_macros)]
macro_rules! sealed {
    (impl$(<$($T:ident),*>)? $trait:ident for $name:ident$(<$($V:ident),*>)? $(where $($rest:tt)*)?) => {
        impl$(<$($T),*>)? $trait for $name$(<$($V),*>)? $(where $($rest)*)? {
            seal!();
        }
    };
}
