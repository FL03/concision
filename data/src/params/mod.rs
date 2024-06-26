/*
   Appellation: params <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # Parameters
//!
//! ## Overview
//!
pub use self::{kinds::*, parameter::*};

pub(crate) mod kinds;
pub(crate) mod parameter;

pub mod store;

mod impls {
    #[cfg(feature = "rand")]
    mod impl_rand;
}

pub trait Param {
    type Key;
    type Value;
}

pub trait Params {
    type Store;
}

pub(crate) mod prelude {
    pub use super::kinds::ParamKind;
    pub use super::parameter::Parameter;
    pub use super::store::ParamStore;
    pub use super::{Param, Params};
}

#[cfg(test)]
mod tests {
    use super::*;
    use concision::linarr;
    use nd::linalg::Dot;
    use nd::prelude::*;

    #[test]
    fn test_parameter() {
        let value = linarr::<f64, Ix2>((3, 3)).unwrap();
        let mut param = Parameter::<f64, Ix2>::new((10, 1), ParamKind::Bias, "bias");
        param.set_params(value.clone());

        assert_eq!(param.kind(), &ParamKind::Bias);
        assert_eq!(param.name(), "bias");

        let x = linarr::<f64, Ix1>((3,)).unwrap();
        assert_eq!(param.dot(&x), value.dot(&x));
    }
}
