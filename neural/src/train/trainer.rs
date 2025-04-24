/*
    Appellation: trainer <module>
    Contrib: @FL03
*/
#![allow(dead_code)]

use crate::Model;

type TrainingConfigMap<T> = std::collections::HashMap<String, T>;

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct TrainingConfig<T = f32> {
    pub(crate) decay: T,
    pub(crate) learning_rate: T,
    pub(crate) momentum: T,
}

pub struct Trainer<'a, M, T>
where
    M: Model<T>,
{
    pub(crate) model: &'a mut M,
    /// the accumulated loss
    pub(crate) loss: T,
}

impl<'a, M, T> Trainer<'a, M, T>
where
    M: Model<T>,
{
    pub fn new(model: &'a mut M) -> Self
    where
        T: Default,
    {
        Self {
            model,
            loss: T::default(),
        }
    }

    pub fn begin(&self) -> &Self {
        self
    }
}
