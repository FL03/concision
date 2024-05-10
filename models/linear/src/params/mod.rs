/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#[doc(inline)]
pub use self::{entry::*, params::*};

pub mod entry;
mod params;

macro_rules! params_ty {
    ($($name:ident<$repr:ident>),* $(,)?) => {
        $(params_ty!(@impl $name<$repr>);)*
    };
    (@impl $name:ident<$repr:ident>) => {
        pub type $name<T = f64, D = nd::Ix2> = ParamsBase<ndarray::$repr<T>, D>;
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
    use nd::Array1;

    #[test]
    fn test_param_kind() {
        for i in [(Param::Bias, "bias"), (Param::Weight, "weight")].iter() {
            let kind = Param::from_str(i.1).unwrap();
            assert_eq!(i.0, kind);
        }
    }

    #[test]
    fn test_ones() {
        let a = LinearParams::<f64>::ones(false, (1, 300)).biased(Array1::ones);
        assert!(a.is_biased());
    }
}
