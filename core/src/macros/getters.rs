/*
    Appellation: getters <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[macro_export]
macro_rules! getters {
    ($($call:ident$(.$field:ident)?<$out:ty>),* $(,)?) => {
        $($crate::getters!(@impl $call$(.$field)?<$out>);)*
    };
    ($via:ident::<[$($call:ident$(.$field:ident)?<$out:ty>),* $(,)?]>) => {
        $($crate::getters!(@impl $via::$call$(.$field)?<$out>);)*
    };
    ($($call:ident$(.$field:ident)?),* $(,)? => $out:ty) => {
        $($crate::getters!(@impl $call$(.$field)?<$out>);)*
    };
    ($via:ident::<[$($call:ident$(.$field:ident)?),* $(,)?]> => $out:ty) => {
        $crate::getters!($via::<[$($call$(.$field)?<$out>),*]>);
    };

    (@impl $call:ident<$out:ty>) => {
        $crate::getters!(@impl $call.$call<$out>);
    };
    (@impl $via:ident::$call:ident<$out:ty>) => {
        $crate::getters!(@impl $via::$call.$call<$out>);
    };
    (@impl $call:ident.$field:ident<$out:ty>) => {
        pub fn $call(&self) -> &$out {
            &self.$field
        }
        paste::paste! {
            pub fn [< $call _mut>](&mut self) -> &mut $out {
                &mut self.$field
            }
        }
    };
    (@impl $via:ident::$call:ident.$field:ident<$out:ty>) => {
        pub fn $call(&self) -> &$out {
            &self.$via.$field
        }
        paste::paste! {
            pub fn [< $call _mut>](&mut self) -> &mut $out {
                &mut self.$via.$field
            }
        }
    };
}
