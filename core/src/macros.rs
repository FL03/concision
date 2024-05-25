/*
   Appellation: macros <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#[macro_use]
mod activate;
#[macro_use]
mod builder;
#[macro_use]
mod enums;
#[macro_use]
mod getters;
#[macro_use]
mod ops;
#[macro_use]
mod toggle;

/// AS
#[macro_export]
macro_rules! dimensional {

    (dim: $name:ident$(())?) => {
        /// Returns a reference to the current dimension, as a slice.
        pub fn as_slice(&self) -> &[usize] {
            self.$name$(())?.shape()
        }

        pub fn into_pattern(self) -> D::Pattern {
            self.$name$(())?.into_pattern()
        }

        pub fn ndim(&self) -> usize {
            self.$name$(())?.ndim()
        }

        pub fn raw_dim(&self) -> D {
            self.$name$(())?.dim().clone()
        }
    };


    ($name:ident) => {
        /// Return the [pattern](ndarray::Dimension::Pattern) of the dimension
        pub fn dim(&self) -> D::Pattern {
            self.$name.dim()
        }
        /// Returns rank (ndim) of the dimension
        pub fn ndim(&self) -> usize {
            self.$name.ndim()
        }
        /// Returns the raw dimension [D](ndarray::Dimension)
        pub fn raw_dim(&self) -> D {
            self.$name.dim()
        }
        /// Returns a reference to the current dimension, as a slice.
        pub fn shape(&self) -> &[usize] {
            self.$name.shape()
        }
    };

    ($name:ident()) => {
        /// Return the [pattern](ndarray::Dimension::Pattern) of the dimension
        pub fn dim(&self) -> D::Pattern {
            self.$name().dim()
        }
        /// Returns rank (ndim) of the dimension
        pub fn ndim(&self) -> usize {
            self.$name().ndim()
        }
        /// Returns the raw dimension [D](ndarray::Dimension)
        pub fn raw_dim(&self) -> D {
            self.$name().raw_dim()
        }
        /// Returns a reference to the current dimension, as a slice.
        pub fn shape(&self) -> &[usize] {
            self.$name().shape()
        }
    };
}
