/*
    Appellation: ops <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[macro_use]
pub(crate) mod toggle;

macro_rules! unary {
    ($($name:ident::$call:ident(self)),* $(,)?) => {
        $(
            unary!(@impl $name::$call(self));
        )*
    };
    ($($name:ident::$call:ident(&self)),* $(,)?) => {
        $(
            unary!(@impl $name::$call(&self));
        )*
    };
    (@branch $name:ident::$call:ident($(&)?self)) => {
        unary!(@impl $name::$call($(&)?self));
    };
    (@impl $name:ident::$call:ident(self)) => {
        pub trait $name {
            type Output;

            fn $call(self) -> Self::Output;
        }
    };
    (@impl $name:ident::$call:ident(&self)) => {
        pub trait $name {
            type Output;

            fn $call(&self) -> Self::Output;
        }
    };
}
#[allow(unused_macros)]
macro_rules! unary_derivative {
    ($($name:ident::$call:ident(self)),* $(,)?) => {
        $(
            unary!(@impl $name::$call(self));
        )*
    };
    ($($name:ident::$call:ident(&self)),* $(,)?) => {
        $(
            unary!(@impl $name::$call(&self));
        )*
    };
    (@branch $name:ident::$call:ident($(&)?self)) => {
        unary!(@impl $name::$call($(&)?self));
    };
    (@impl $name:ident::$call:ident(self)) => {
        paste::paste! {
            pub trait $name {
                type Output;

                fn $call(self) -> Self::Output;

                fn [<$call _derivative>](self) -> Self::Output;
            }
        }
    };
    (@impl $name:ident::$call:ident(&self)) => {
        pub trait $name {
            type Output;

            fn $call(&self) -> Self::Output;
        }
    };
}
