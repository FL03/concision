use crate::dataset::Dataset;

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
