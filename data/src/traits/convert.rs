/*
    Appellation: convert <module>
    Created At: 2026.01.13:14:06:50
    Contrib: @FL03
*/
use crate::dataset::DatasetBase;
/// The [`AsDataset`] trait defines the conversion from some reference into a dataset.
pub trait AsDataset<U, V> {
    fn as_dataset(&self) -> DatasetBase<U, V>;
}
/// Thge [`IntoDataset`] trait defines a method for consuming the caller to convert it into a
/// dataset.
pub trait IntoDataset<U, V> {
    fn into_dataset(self) -> DatasetBase<U, V>;
}

/*
 ************* Implementations *************
*/
impl<U, V, A> AsDataset<U, V> for A
where
    A: AsRef<DatasetBase<U, V>>,
    U: Clone,
    V: Clone,
{
    fn as_dataset(&self) -> DatasetBase<U, V> {
        self.as_ref().clone()
    }
}

impl<U, V, A> IntoDataset<U, V> for A
where
    A: Into<DatasetBase<U, V>>,
{
    fn into_dataset(self) -> DatasetBase<U, V> {
        self.into()
    }
}
