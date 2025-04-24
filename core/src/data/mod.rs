/*
    Appellation: data <module>
    Contrib: @FL03
*/
//! this module implements a dataset abstraction for machine learning tasks.
//!
//!
#[doc(inline)]
pub use self::dataset::Dataset;

pub mod dataset;

#[allow(unused_imports)]
pub(crate) mod prelude {
    pub use super::dataset::Dataset;
}

pub trait DataPoint {
    type Data;
    type Label;
    fn data(&self) -> &Self::Data;
    fn label(&self) -> &Self::Label;
}

pub trait Records {
    type Inputs;
    type Targets;

    fn inputs(&self) -> &Self::Inputs;

    fn inputs_mut(&mut self) -> &mut Self::Inputs;

    fn targets(&self) -> &Self::Targets;

    fn targets_mut(&mut self) -> &mut Self::Targets;
}

pub trait AsDataset<U, V> {
    fn as_dataset(&self) -> Dataset<U, V>;
}
pub trait IntoDataset<U, V> {
    fn into_dataset(self) -> Dataset<U, V>;
}

/*
 ************* Implementations *************
*/
impl<U, V, A> AsDataset<U, V> for A
where
    A: AsRef<Dataset<U, V>>,
    U: Clone,
    V: Clone,
{
    fn as_dataset(&self) -> Dataset<U, V> {
        self.as_ref().clone()
    }
}
impl<U, V, A> IntoDataset<U, V> for A
where
    A: Into<Dataset<U, V>>,
{
    fn into_dataset(self) -> Dataset<U, V> {
        self.into()
    }
}
