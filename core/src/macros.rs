/*
   Appellation: macros <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

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
