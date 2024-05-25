/*
    Appellation: enums <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! from_variant {
    ($base:ident::$variant:ident $($rest:tt)*) => {
        from_variant!(@branch $base::$variant $($rest)*);
    };
    (@branch $base:ident::$variant:ident($from:ty)$(.$method:ident())*) => {
        from_variant!(@impl $base::$variant($from)$(.$method())*);
    };
    (@branch $base:ident::$variant:ident{$(<$err:ty>$(.$method:ident())*),* $(,)?}) => {
        $(
            from_variant!(@impl $base::$variant($err)$(.$method())*);
        )*
    };
    (@impl $base:ident::$variant:ident($from:ty)$(.$method:ident())*) => {
        impl From<$from> for $base {
            fn from(val: $from) -> Self {
                Self::$variant(val$(.$method())*)
            }
        }
    };
}

#[allow(unused_macros)]
macro_rules! nested_enum_constructor {
    ($variant:ident<$inner:ident>, $method:ident, [$($call:ident),*]) => {
        nested_enum_constructor!(@loop $variant<$inner>, $method, [$($call),*]);
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
