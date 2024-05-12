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

macro_rules! ndbuilder {
    ($method:ident $($rest:tt)*) => {
        ndbuilder!(@impl $method $($rest)*);
    };
    (@impl $method:ident where $($rest:tt)*) => {
        pub fn $method<Sh>(shape: Sh) -> Self
        where
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            Self::builder(shape, ArrayBase::$method)
        }
    };
    (@impl $method:ident.$call:ident where $($rest:tt)*) => {
        pub fn $method<Sh>(shape: Sh) -> Self
        where
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            Self::builder(shape, ArrayBase::$call)
        }
    };
}

macro_rules! param_views {
    ($method:ident::$($rest:tt)*) => {
        param_views!(@impl $method.$method::$($rest)*);
    };
    ($method:ident.$call:ident::$($rest:tt)*) => {
        param_views!(@impl $method.$call::$($rest)*);
    };
    (@impl $method:ident.$call:ident::<$view:ident>(self) where $($rest:tt)*) => {
        pub fn $method(self) -> $crate::params::ParamsBase<$view<A>, D>
        where
            $($rest)*
        {
            param_views!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(mut self) where $($rest:tt)*) => {
        pub fn $method(mut self) -> $crate::params::ParamsBase<$view<A>, D>
        where
            $($rest)*
        {
            param_views!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&self) where $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::ParamsBase<$view<A>, D>
        where
            $($rest)*
        {
            param_views!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&mut self) where $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::ParamsBase<$view<A>, D>
        where
            $($rest)*
        {
            param_views!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&self) where $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::ParamsBase<$view<&'_ A>, D>
        where
            $($rest)*
        {
            param_views!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&mut self) where $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::ParamsBase<$view<&'_ mut A>, D>
        where
            $($rest)*
        {
            param_views!(@apply $call(self))
        }
    };
    (@apply $call:ident($self:expr)) => {
        $crate::params::ParamsBase {
            q: $self.q.$call(),
            k: $self.k.$call(),
            v: $self.v.$call(),
        }
    };
}
