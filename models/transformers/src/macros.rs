/*
    Appellation: macros <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! ndbuilder {
    ($method:ident$(::$call:ident)?() where $($rest:tt)*) => {
        ndbuilder!(@impl $method$(::$call)?() where $($rest)*);
    };
    (@impl $method:ident() where $($rest:tt)*) => {
        ndbuilder!(@impl $method::$method() where $($rest)*);
    };
    (@impl $method:ident::$call:ident() where $($rest:tt)*) => {
        pub fn $method<Sh: ndarray::ShapeBuilder<Dim = D>>(shape: Sh) -> Self where $($rest)* {
            Self::builder(shape, ndarray::ArrayBase::$call)
        }
    };
}

// # TODO:
macro_rules! ndview {
    ($method:ident::$($rest:tt)*) => {
        ndview!(@impl $method.$method::$($rest)*);
    };
    ($method:ident.$call:ident::$($rest:tt)*) => {
        ndview!(@impl $method.$call::$($rest)*);
    };
    (@impl $method:ident.$call:ident::<$view:ident>(self) where $($rest:tt)*) => {
        pub fn $method(self) -> $crate::params::QkvBase<$view<A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(mut self) where $($rest:tt)*) => {
        pub fn $method(mut self) -> $crate::params::QkvBase<$view<A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&self) where $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::QkvBase<$view<A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&mut self) where $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::QkvBase<$view<A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&self) where $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::QkvBase<$view<&'_ A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&mut self) where $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::QkvBase<$view<&'_ mut A>, D>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@apply $call:ident($self:expr)) => {
        $crate::params::QkvBase {
            q: $self.q.$call(),
            k: $self.k.$call(),
            v: $self.v.$call(),
        }
    };
}
