/*
    Appellation: data <module>
    Contrib: @FL03
*/
//! this module implements a dataset abstraction for machine learning tasks.
//!
//!
#[doc(inline)]
pub use self::dataset::DatasetBase;

pub mod dataset;

#[allow(unused_imports)]
pub(crate) mod prelude {
    pub use super::dataset::DatasetBase;
}

pub trait DataPoint {
    type Data;
    type Label;
    fn data(&self) -> &Self::Data;
    fn label(&self) -> &Self::Label;
}
