/*
    Appellation: dataset <module>
    Contrib: @FL03
*/
#![allow(dead_code)]

pub struct DatasetBase<R, T> {
    pub(crate) records: R,
    pub(crate) targets: T,
}
