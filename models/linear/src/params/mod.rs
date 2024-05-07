/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{entry::*, params::*};

pub mod entry;
mod params;

mod impls {
    mod impl_params;
    #[cfg(feature = "rand")]
    mod impl_rand;
    mod impl_serde;
}

macro_rules! params_ty {
    ($($name:ident<$repr:ident>),* $(,)?) => {
        $(params_ty!(@impl $name<$repr>);)*
    };
    (@impl $name:ident<$repr:ident>) => {
        pub type $name<T = f64, D = nd::Ix2> = LinearParamsBase<ndarray::$repr<T>, D>;
    };
}

params_ty!(LinearParams<OwnedRepr>, LinearParamsShared<OwnedArcRepr>,);

pub(crate) mod prelude {
    pub use super::entry::{Entry as LinearEntry, Param as LinearParam};
    pub use super::LinearParams;
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
        let a = LinearParams::<f64>::ones(false, (1, 300)).biased(nd::Array1::ones);

        assert!(a.is_biased());
    }
}
