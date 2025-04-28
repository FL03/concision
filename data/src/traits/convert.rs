use crate::dataset::DatasetBase;

pub trait AsDataset<U, V> {
    fn as_dataset(&self) -> DatasetBase<U, V>;
}
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
