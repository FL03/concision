/*
   Appellation: macros <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
#![allow(unused_macros)]

macro_rules! error_from {
    ($base:ident::$variant:ident<$($err:ty),* $(,)?>) => {
        error_from!(@loop $base::$variant<$($err),*>);
    };
    ($base:ident::$variant:ident<$err:ty>$($rest:tt)*) => {
        error_from!(@loop $base::$variant<$($err),*>$($rest)*);
    };
    (@loop $base:ident::$variant:ident<$($err:ty),* $(,)?>) => {
        $(
            error_from!(@impl $base::$variant<$err>);
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
    ($($rest:tt),* $(,)?) => {
        $(
            variant_constructor!(@loop $($rest),*);
        )*
    };
    ($($variant:ident::$method:ident$(($call:expr))?),* $(,)?) => {
        $(
            variant_constructor!(@loop $variant::$method$(($call))?);
        )*
    };

    (@loop $variant:ident::$method:ident$(($call:expr))?) => {
        pub fn $method() -> Self {
            Self::$variant$(($call))?
        }
    };
}

macro_rules! impl_unary {
    ($name:ident::$call:ident<$T:ty>($f:expr) $($rest:tt)*) => {
        impl_unary!(@impl $name::$call<$T>($f) $($rest)*);
    };
    (@impl $name:ident::$call:ident<$T:ty>($f:expr)) => {
        impl $name for $T {
            type Output = $T;

            fn $call(&self) -> Self::Output {
                $f(self)
            }
        }
    };
}

macro_rules! unary {
    ($($name:ident::$call:ident),* $(,)?) => {
        $(
            unary!(@impl $name::$call(self));
        )*
    };
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

#[macro_export]
macro_rules! builder {
    ($(#[derive($($d:ident),+)])?$name:ident::<$inner:ty> {$($k:ident: $v:ty),* $(,)?}) => {
        $crate::builder!(@loop builder: $name, derive: [$($($d),+)?], inner: $inner {$($k: $v),*});
    };
    ($(#[derive($($d:ident),+)])? $name:ident($inner:ty) {$($k:ident: $v:ty),* $(,)?}) => {
        $crate::builder!(@loop builder: $name, derive: [$($($d),+)?], inner: $inner {$($k: $v),*});
    };
    (@loop builder: $name:ident, derive: [$($d:ident),* $(,)?], inner: $inner:ty {$($k:ident: $v:ty),* $(,)?}) => {

        #[derive(Default, $($d),*)]
        pub struct $name {
            inner: $inner,
        }

        $crate::builder!(@impl builder: $name, inner: $inner {$($k: $v),*});
    };
    (@impl builder: $name:ident, inner: $inner:ty {$($k:ident: $v:ty),* $(,)?}) => {
        impl $name {
            pub fn new() -> Self {
                Self {
                    inner: Default::default()
                }
            }

            pub fn from_inner(inner: $inner) -> Self {
                Self { inner }
            }

            pub fn build(self) -> $inner {
                self.inner
            }

            $(
                pub fn $k(mut self, $k: $v) -> Self {
                    self.inner.$k = $k;
                    self
                }
            )*
        }
    };
}

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

/// AS
#[macro_export]
macro_rules! dimensional {

    (dim: $name:ident$(())?) => {
        /// Returns a reference to the current dimension, as a slice.
        pub fn as_slice(&self) -> &[usize] {
            self.$name$(())?.shape()
        }

        pub fn into_pattern(self) -> D::Pattern {
            self.$name$(())?.into_pattern()
        }

        pub fn ndim(&self) -> usize {
            self.$name$(())?.ndim()
        }

        pub fn raw_dim(&self) -> D {
            self.$name$(())?.dim().clone()
        }
    };


    ($name:ident) => {
        /// Return the [pattern](ndarray::Dimension::Pattern) of the dimension
        pub fn dim(&self) -> D::Pattern {
            self.$name.dim()
        }
        /// Returns rank (ndim) of the dimension
        pub fn ndim(&self) -> usize {
            self.$name.ndim()
        }
        /// Returns the raw dimension [D](ndarray::Dimension)
        pub fn raw_dim(&self) -> D {
            self.$name.dim()
        }
        /// Returns a reference to the current dimension, as a slice.
        pub fn shape(&self) -> &[usize] {
            self.$name.shape()
        }
    };

    ($name:ident()) => {
        /// Return the [pattern](ndarray::Dimension::Pattern) of the dimension
        pub fn dim(&self) -> D::Pattern {
            self.$name().dim()
        }
        /// Returns rank (ndim) of the dimension
        pub fn ndim(&self) -> usize {
            self.$name().ndim()
        }
        /// Returns the raw dimension [D](ndarray::Dimension)
        pub fn raw_dim(&self) -> D {
            self.$name().raw_dim()
        }
        /// Returns a reference to the current dimension, as a slice.
        pub fn shape(&self) -> &[usize] {
            self.$name().shape()
        }
    };
}
