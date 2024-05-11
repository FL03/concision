/*
    Appellation: macros <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! access {
    ($($var:ident),* $(,)?) => {
        $(access!(@impl $var);)*
    };
    ($via:ident::<$($var:ident),* $(,)?>) => {
        $(access!(@impl $via::$var);)*
    };
    (@impl $var:ident) => {
        pub fn $var(&self) -> &ArrayBase<S, D> {
            &self.$var
        }
        paste::paste! {
            pub fn [< $var _mut>](&mut self) -> &mut ArrayBase<S, D> {
                &mut self.$var
            }
        }
    };
    (@impl $via:ident::$var:ident) => {
        pub fn $var(&self) -> &ArrayBase<S, D> {
            &self.$via.$var
        }
        paste::paste! {
            pub fn [< $var _mut>](&mut self) -> &mut ArrayBase<S, D> {
                &mut self.$via.$var
            }
        }
    };
}

macro_rules! fwd_builder {
    ($method:ident.$call:ident where $($rest:tt)*) => {
        pub fn $method<Sh>(shape: Sh) -> Self
        where
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            Self::builder(shape, ArrayBase::$call)
        }
    };
}

macro_rules! qkv_builder {

    ($method:ident.$call:ident where $($rest:tt)*) => {
        pub fn $method<Sh>(shape: Sh) -> Self
        where
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            Self::builder(shape, ArrayBase::$call)
        }
    };
}
