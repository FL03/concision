/*
    Appellation: macros <module>
    Contrib: @FL03
*/

#[macro_use]
pub(crate) mod gsw;

macro_rules! unary_op_trait {
    ($($name:ident::$call:ident($($rest:tt)*)),* $(,)?) => {
        $(
            unary!(@impl $name::$call($($rest)*));
        )*
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
        paste::paste! {
            pub trait $name {
                type Output;

                fn $call(&self) -> Self::Output;

                fn [<$call _derivative>](&self) -> Self::Output;
            }
        }
    };
}

macro_rules! impl_unary_op_trait {
    ($name:ident::<$T:ty, Output = $out:ty>::$call:ident(self) => {
        f: $func:expr,
        df: $df:expr,
    }) => {
        paste::paste! {
            impl $name for $T {
                type Output = $out;

                fn $call(self) -> Self::Output {
                    $func(self)
                }

                fn [<$call _derivative>](self) -> Self::Output {
                    $df(self)
                }
            }
        }
    };

    ($name:ident::$call:ident(&self) => {
        f: $func:expr,
        df: $df:expr,
    }) => {
        paste::paste! {
            impl $name for f32 {
                type Output = f32;

                fn $call(&self) -> Self::Output {
                    $func(self)
                }

                fn [<$call _derivative>](&self) -> Self::Output {
                    $df(self)
                }
            }
        }
    };
}