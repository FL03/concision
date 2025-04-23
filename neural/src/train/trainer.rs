/*
    Appellation: trainer <module>
    Contrib: @FL03
*/
#![allow(dead_code)]

pub struct Trainer<'a, M, T> {
    pub(crate) model: &'a mut M,
    pub(crate) learning_rate: T,
    pub(crate) momentum: T,
    pub(crate) decay: T,
    pub(crate) batch_size: usize,
    pub(crate) epochs: usize,
    pub(crate) loss: T,
}
