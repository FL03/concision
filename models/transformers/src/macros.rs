/*
    Appellation: macros <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[macro_use]
mod params;

macro_rules! ndbuilder {
    ($method:ident$(::$call:ident)?() $($where:tt)*) => {
        ndbuilder!(@impl $method$(::$call)?() $($where)*);
    };
    (@impl $method:ident() $($where:tt)*) => {
        ndbuilder!(@impl $method::$method() $($where)*);
    };
    (@impl $method:ident::$call:ident() $($where:tt)*) => {
        pub fn $method<Sh: ndarray::ShapeBuilder<Dim = D>>(shape: Sh) -> Self $($where)* {
            Self::builder(shape, ndarray::ArrayBase::$call)
        }
    };
}