/*
    Appellation: trainer <module>
    Contrib: @FL03
*/
#![allow(dead_code)]

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct TrainingConfig<T = f32> {
    pub(crate) learning_rate: T,
    pub(crate) momentum: T,
    pub(crate) decay: T,
    pub(crate) batch_size: usize,
    pub(crate) epochs: usize,
}

pub struct Trainer<'a, M, T> {
    pub(crate) model: &'a mut M,
    pub(crate) config: TrainingConfig<T>,
    /// the accumulated loss
    pub(crate) loss: T,
}
