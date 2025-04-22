/*
    Appellation: dataset <module>
    Contrib: @FL03
*/

pub struct DatasetBase<R, T> {
    pub(crate) records: R,
    pub(crate) targets: T,
}
