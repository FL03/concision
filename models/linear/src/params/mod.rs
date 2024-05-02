/*
    Appellation: params <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{entry::*, params::*};

pub mod entry;
mod params;

mod impls {
    mod impl_params;
    mod impl_rand;
    mod impl_serde;
}

pub(crate) mod prelude {
    pub use super::entry::{Entry as LinearEntry, Param as LinearParam};
    pub use super::params::LinearParams;
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
}
