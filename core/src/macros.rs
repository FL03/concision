/*
   Appellation: macros <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![allow(unused_macros)]

macro_rules! impl_from_error {
    ($base:ident::$variant:ident<$($err:ty),* $(,)?>) => {
        impl_from_error!(@loop $base::$variant<$($err),*>);
    };
    ($base:ident::$variant:ident<$err:ty>$($rest:tt)*) => {
        impl_from_error!(@loop $base::$variant<$($err),*>$($rest)*);
    };
    (@loop $base:ident::$variant:ident<$($err:ty),* $(,)?>) => {
        $(
            impl_from_error!(@impl $base::$variant<$err>);
        )*
    };
    (@impl $base:ident::$variant:ident<$err:ty>) => {
        impl From<$err> for $base {
            fn from(err: $err) -> Self {
                Self::$variant(err.to_string())
            }
        }
    };
    (@impl $base:ident::$variant:ident<$err:ty>.$method:ident) => {
        impl From<$err> for $base {
            fn from(err: $err) -> Self {
                Self::$variant(err.$method())
            }
        }
    };
}

macro_rules! nested_constructor {
    ($variant:ident<$inner:ident>, $method:ident, [$($call:ident),*]) => {
        nested_constructor!(@loop $variant<$inner>, $method, [$($call),*]);
    };
    (@loop $variant:ident<$inner:ident>, $method:ident, [$($call:ident),*]) => {
        pub fn $method(inner:$inner) -> Self {
            Self::$variant(inner)
        }

        $(
            pub fn $call() -> Self {
                Self::$method($inner::$call())
            }
        )*

    };
}

macro_rules! variant_constructor {
    ($(($($rest:tt),*)),*) => {
        $(
            variant_constructor!(@loop $($rest),*);
        )*
    };
    ($(($variant:ident $($rest:tt),*, $method:ident)),*) => {
        $(
            variant_constructor!(@loop $variant $($rest),*, $method);
        )*
    };
    (@loop $variant:ident, $method:ident) => {
        pub fn $method() -> Self {
            Self::$variant
        }
    };

    (@loop $variant:ident($call:expr), $method:ident) => {
        pub fn $method() -> Self {
            Self::$variant($call())
        }
    };


}