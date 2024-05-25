/*
    Appellation: multi_head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Config;
use crate::AttentionHead;
use linear::{Biased, Linear};
use nd::prelude::*;
use nd::{DataOwned, OwnedRepr, RawData};

pub struct MultiHeadAttention<A = f64, D = Ix2, S = OwnedRepr<A>>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub(crate) config: Config,
    pub(crate) head: AttentionHead<A, D, S>,
    pub(crate) linears: Vec<Linear<A, Biased, D, S>>,
}

impl<A, S, D> MultiHeadAttention<A, D, S>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub const fn config(&self) -> &Config {
        &self.config
    }

    pub const fn head(&self) -> &AttentionHead<A, D, S> {
        &self.head
    }

    pub fn head_mut(&mut self) -> &mut AttentionHead<A, D, S> {
        &mut self.head
    }

    pub fn linears(&self) -> &[Linear<A, Biased, D, S>] {
        &self.linears
    }
}

impl<A, S> MultiHeadAttention<A, Ix2, S>
where
    S: RawData<Elem = A>,
{
    pub fn std(d_model: usize, heads: usize) -> Self
    where
        A: Clone + Default,
        S: DataOwned,
    {
        let config = Config::new().d_model(d_model).heads(heads).build();
        let linears = (0..4)
            .map(|_| Linear::from_features(d_model, d_model))
            .collect();
        Self {
            config,
            head: AttentionHead::std(d_model, config.dk()),
            linears,
        }
    }
}

impl<A, S, D> Default for MultiHeadAttention<A, D, S>
where
    A: Default,
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    fn default() -> Self {
        Self {
            config: Config::default(),
            head: AttentionHead::default(),
            linears: Vec::new(),
        }
    }
}
