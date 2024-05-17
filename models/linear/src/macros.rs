/*
    Appellation: macros <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! impl_params_builder {
    ($method:ident$(.$call:ident)? where $($rest:tt)*) => {
        impl_params_builder!(@impl $method$(.$call)? where $($rest)*);
    };
    (@impl $method:ident where $($rest:tt)*) => {
        impl_params_builder!(@impl $method.$method where $($rest)*);
    };
    (@impl $method:ident.$call:ident where $($rest:tt)*) => {
        pub fn $method<Sh>(shape: Sh) -> Self
        where
            K: $crate::params::mode::ParamMode,
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            let dim = shape.into_shape().raw_dim().clone();
            ParamsBase {
                bias: build_bias(K::BIASED, dim.clone(), |dim| ndarray::ArrayBase::$call(dim)),
                weight: ndarray::ArrayBase::$call(dim),
                _mode: ::core::marker::PhantomData::<K>,
            }
        }
    };
}

macro_rules! impl_model_builder {
    ($method:ident$(.$call:ident)? where $($rest:tt)*) => {
        impl_model_builder!(@impl $method$(.$call)? where $($rest)*);
    };
    (@impl $method:ident where $($rest:tt)*) => {
        impl_model_builder!(@impl $method.$method where $($rest)*);
    };
    (@impl $method:ident.$call:ident where $($rest:tt)*) => {
        pub fn $method<Sh>(shape: Sh) -> Self
        where
            K: $crate::params::mode::ParamMode,
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            let dim = shape.into_shape().raw_dim().clone();
            $crate::model::Linear {
                config: $crate::model::Config::<K, D>::new().with_shape(dim.clone()),
                params: $crate::params::ParamsBase::$call(dim),
            }
        }
    };
}

macro_rules! ndview {
    ($method:ident::$($rest:tt)*) => {
        ndview!(@impl $method.$method::$($rest)*);
    };
    ($method:ident.$call:ident::$($rest:tt)*) => {
        ndview!(@impl $method.$call::$($rest)*);
    };
    (@impl $method:ident.$call:ident::<$view:ident>(self) where $($rest:tt)*) => {
        pub fn $method(self) -> $crate::params::ParamsBase<$view<A>, D, K>
        where
            $($rest)*
        {
            ndview!(@apply $call(self))
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(mut self) where $($rest:tt)*) => {
        pub fn $method(mut self) -> $crate::params::ParamsBase<$view<A>, D, K>
        where
            $($rest)*
        {
            ndview!(@apply $call(self).as_mut())
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&self) where $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::ParamsBase<$view<A>, D, K>
        where
            $($rest)*
        {
            ndview!(@apply $call(self).as_ref())
        }
    };
    (@impl $method:ident.$call:ident::<$view:ident>(&mut self) where $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::ParamsBase<$view<A>, D, K>
        where
            $($rest)*
        {
            ndview!(@apply $call(self).as_mut())
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&self) where $($rest:tt)*) => {
        pub fn $method(&self) -> $crate::params::ParamsBase<$view<&'_ A>, D, K>
        where
            $($rest)*
        {
            ndview!(@apply $call(&self).as_ref())
        }
    };
    (@impl $method:ident.$call:ident::<'a, $view:ident>(&mut self) where $($rest:tt)*) => {
        pub fn $method(&mut self) -> $crate::params::ParamsBase<$view<&'_ mut A>, D, K>
        where
            $($rest)*
        {
            ndview!(@apply $call(self).as_mut())
        }
    };
    (@apply $call:ident($self:expr)$(.$as:ident())?) => {
        $crate::params::ParamsBase {
            bias: $self.bias$(.$as())?.map(|arr| arr.$call()),
            weight: $self.weight.$call(),
            _mode: $self._mode,
        }
    };
}
