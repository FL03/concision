/*
   Appellation: concision-macros <library>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Concision Macros

#[macro_export]
macro_rules! linspace {
    ( $x:expr ) => {
        {
            let dim = $x.into_dimension();
            let n = $dim.as_array_view().product();
            ndarray::Array::linspace(T::one(), T::from(n).unwrap(), n).into_shape(dim).unwrap()
        }
    };
    ( $( $x:expr ),* ) => {
        {
            let mut res = Vec::new();
            $(
                res.push(linarr!($x));
            )*
            res
        }
    };
}
