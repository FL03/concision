/*
    Appellation: errors <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! from_err {
    ($S:ty: $($($p:ident)::*($err:ty)),* $(,)*) => {
        $(
            from_err!(@impl impl $S: $err => $($p)::*);
        )*
    };
    (@impl impl $S:ty: $err:ty => $($p:ident)::*) => {
        impl From<$err> for $S {
            fn from(err: $err) -> Self {
                $($p)::*(err)
            }
        }
    };
    (@impl impl<T> $S:ty: $err:ty => $($p:ident)::*) => {
        impl<T> From<$err> for $S {
            fn from(err: $err) -> Self {
                $($p)::*(err)
            }
        }
    };
}

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

macro_rules! err {
    ($name:ident $($rest:tt)*) => {
        err!(@base $name $($rest)*);
    };
    (@base $name:ident {$($rest:tt)*}) => {
        #[derive(
            Clone,
            Copy,
            Debug,
            Eq,
            Hash,
            Ord,
            PartialEq,
            PartialOrd,
            scsys::VariantConstructors,
            strum::AsRefStr,
            strum::Display,
            strum::EnumCount,
            strum::EnumIs,
            strum::EnumIter,
            strum::EnumString,
            strum::VariantNames,
        )]
        #[cfg_attr(
            feature = "serde",
            derive(serde::Deserialize, serde::Serialize),
            serde(rename_all = "snake_case", untagged)
        )]
        #[strum(serialize_all = "snake_case")]
        pub enum $name {
            $($rest)*
        }
    };
}
