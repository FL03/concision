/*
    Appellation: data <module>
    Contrib: @FL03
*/
//! Standard datasets to train the models on.
#[doc(inline)]
pub use self::dataset::Dataset;

pub mod dataset;

pub(crate) mod prelude {
    pub use super::dataset::Dataset;
    pub use super::{AsDataset, IntoDataset, Records};
}

pub trait DataPoint {
    type Data;
    type Label;
    fn data(&self) -> &Self::Data;
    fn label(&self) -> &Self::Label;
}

/// This trait generically defines the basic type of dataset that can be used throughout the 
/// framework.
pub trait Records {
    type Inputs;
    type Targets;

    fn inputs(&self) -> &Self::Inputs;

    fn inputs_mut(&mut self) -> &mut Self::Inputs;

    fn targets(&self) -> &Self::Targets;

    fn targets_mut(&mut self) -> &mut Self::Targets;

    fn set_inputs(&mut self, inputs: Self::Inputs) {
        *self.inputs_mut() = inputs;
    }

    fn set_targets(&mut self, targets: Self::Targets) {
        *self.targets_mut() = targets;
    }
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
