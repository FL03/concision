/*
    Appellation: macros <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[allow(unused_macros)]
macro_rules! params {
    {$($k:ident: $v:expr),* $(,)?} => {
        params!(@new $($k: $v),*);
    };
    (@new bias: $b:expr, weights: $w:expr, mode: $mode:ty) => {
        $crate::params::ParamsBase {
            bias: $b,
            weights: $w,
            _mode: core::marker::PhantomData::<$mode>,
        }
    };
    (@new bias: $b:expr, weights: $w:expr) => {
        params!(@new bias: $b, weights: $w, mode: $crate::params::mode::Biased);
    };
    (@new bias: $b:expr, weights: $w:expr) => {
        params!(@new bias: Some($b), weights: $w, mode: $crate::params::mode::Biased);
    };
    (@new weights: $w:expr) => {
        params!(@new bias: None, weights: $w, mode: $crate::params::mode::Unbiased);
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
