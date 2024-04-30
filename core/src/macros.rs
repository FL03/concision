/*
   Appellation: macros <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! impl_from_error {
    ($base:ident::$variant:ident<$($err:ty),* $(,)?>) => {
        $(
            impl_from_error!(@impl $base::$variant<$err>);
        )*
    };
    ($base:ident::$variant:ident($p:path)<$($err:ty),* $(,)?>) => {
        $(
            impl_from_error!(@impl $base::$variant($p)<$err>);
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