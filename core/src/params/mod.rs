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
    pub use super::Param;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linarr;
    use ndarray::linalg::Dot;
    use ndarray::prelude::{Ix1, Ix2};

    #[test]
    fn test_parameter() {
        let a = linarr::<f64, Ix1>((3,)).unwrap();
        let p = linarr::<f64, Ix2>((3, 3)).unwrap();
        let mut param = Parameter::<f64, Ix2>::new((10, 1), ParamKind::Bias, "bias");
        param.set_params(p.clone());

        assert_eq!(param.kind(), &ParamKind::Bias);
        assert_eq!(param.name(), "bias");
        assert_eq!(param.dot(&a), p.dot(&a));
    }
}
