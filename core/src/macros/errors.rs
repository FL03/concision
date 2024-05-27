/*
    Appellation: errors <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/


macro_rules! impl_err {
    ($($ty:ty),* $(,)*) => {
        $(impl_err!(@impl $ty);)*
    };
    (@impl $ty:ty) => {
        impl $crate::error::ErrorKind for $ty {}

        #[cfg(feature = "std")]
        impl std::error::Error for $ty {}
    };
}

#[allow(unused_macros)]
macro_rules! err_from {
    ($($ty:ty),* $(,)*) => {
        $(err_from!(@impl $ty);)*
    };
    (@impl $ty:ty) => {

    };
}