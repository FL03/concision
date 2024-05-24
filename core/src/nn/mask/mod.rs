/*
    Appellation: mask <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::mask::*;

pub(crate) mod mask;

pub(crate) mod prelude {
    pub use super::mask::Mask;
    pub use super::NdMask;
}

use nd::{ArrayBase, Dimension, Ix2, RawData};

pub trait NdMask<D = Ix2>
where
    D: Dimension,
{
    type Data: RawData<Elem = bool>;
}

impl<S, D> NdMask<D> for ArrayBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = bool>,
{
    type Data = S;
}
