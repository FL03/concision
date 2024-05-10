/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#[doc(inline)]
pub use self::entry::{Entry, Param};
pub use self::mode::{Biased, ParamMode, Unbiased};
pub use self::params::ParamsBase;

mod params;

pub mod entry;
pub mod mode;

macro_rules! params_ty {
    ($($name:ident<$repr:ident>),* $(,)?) => {
        $(params_ty!(@impl $name<$repr>);)*
    };
    (@impl $name:ident<$repr:ident>) => {
        pub type $name<K = Biased, T = f64, D = nd::Ix2> = ParamsBase<ndarray::$repr<T>, D, K>;
    };
}

params_ty!(LinearParams<OwnedRepr>, LinearParamsShared<OwnedArcRepr>,);

pub(crate) mod prelude {
    pub use super::{LinearParams, LinearParamsShared};
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::str::FromStr;

    #[test]
    fn test_param_kind() {
        for i in [(Param::Bias, "bias"), (Param::Weight, "weight")].iter() {
            let kind = Param::from_str(i.1).unwrap();
            assert_eq!(i.0, kind);
        }
    }

    #[test]
    fn test_ones() {
        let a = LinearParams::<Biased, f64>::ones((1, 300));
        assert!(a.is_biased());
    }
}
