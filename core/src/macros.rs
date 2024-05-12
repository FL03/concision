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
    ($($variant:ident.$method:ident$(($call:expr))?),* $(,)?) => {
        $(
            variant_constructor!(@loop $variant.$method$(($call))?);
        )*
    };

    (@loop $variant:ident.$method:ident$(($call:expr))?) => {
        pub fn $method() -> Self {
            Self::$variant$(($call))?
        }
    };
}

macro_rules! impl_unary {
    ($name:ident.$call:ident<$T:ty>($f:expr) $($rest:tt)*) => {
        impl_unary!(@impl $name.$call<$T>($f) $($rest)*);
    };
    (@impl $name:ident.$call:ident<$T:ty>($f:expr)) => {
        impl $name for $T {
            type Output = $T;

            fn $call(&self) -> Self::Output {
                $f(self)
            }
        }
    };
}

macro_rules! build_unary_trait {
    ($($name:ident.$call:ident),* $(,)?) => {
        $(
            build_unary_trait!(@impl $name.$call);
        )*
    };
    (@impl $name:ident.$call:ident) => {
        pub trait $name {
            type Output;

            fn $call(&self) -> Self::Output;
        }
    };
}

#[macro_export]
macro_rules! builder {
    ($(#[derive($($d:ident),*)])?$name:ident::<$inner:ty> {$($k:ident: $v:ty),* $(,)?}) => {
        $crate::builder!(@loop $(#[derive($($d),*)])?$name::<$inner> {$($k: $v),*});
    };
    (@loop #[derive($($d:ident),*)] $name:ident::<$inner:ty> {$($k:ident: $v:ty),* $(,)?}) => {
        pub struct $name {
            inner: $inner,
        }

        impl $name {
            pub fn new() -> Self {
                Self { inner: Default::default() }
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
    (@loop $name:ident::<$inner:ty> {$($k:ident: $v:ty),* $(,)?}) => {
        pub struct $name {
            inner: $inner,
        }

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
