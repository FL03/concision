/*
    Appellation: macros <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[allow(unused_macros)]
macro_rules! params {
    (bias: $bias:expr, weight: $weight:expr $(,)?) => {
        params!(@biased bias: $bias, weight: $weight);
    };
    (@new bias: $bias:expr, weight: $weight:expr, mode: $mode:ty) => {
        $crate::params::ParamsBase {
            bias: $bias,
            weights: $weight,
            _mode: $mode,
        }
    };
    (@biased bias: $bias:expr, weight: $weight:expr, mode: $mode:ty) => {
        $crate::params::ParamsBase {
            bias: $bias,
            weights: $weight,
            _mode: core::marker::PhantomData,
        }
    };
}


macro_rules! impl_param_builder {
    ($call:ident where $($rest:tt)*) => {
        impl_param_builder!(@impl $call where $($rest)*);
    };
    (@impl $call:ident where $($rest:tt)*) => {
        pub fn $call<Sh>(shape: Sh) -> Self
        where
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            let shape = shape.into_shape();
            let dim = shape.raw_dim().clone();
            ParamsBase {
                bias: build_bias(K::BIASED, dim.clone(), |dim| ndarray::ArrayBase::$call(dim)),
                weights: ndarray::ArrayBase::$call(dim),
                _mode: core::marker::PhantomData,
            }
        }
    };
}
