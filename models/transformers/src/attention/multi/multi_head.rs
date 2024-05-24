/*
    Appellation: multi_head <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Config;
use crate::AttentionHead;
use linear::{Biased, Linear};
use nd::prelude::*;


#[derive(Default)]
pub struct MultiHeadAttention<A = f64, D = Ix2> where D: Dimension, {
    pub(crate) attention: Option<AttentionHead<A, D>>,
    pub(crate) config: Config,
    pub(crate) linears: Vec<Linear<A, Biased, D>>,
}

impl<A, D> MultiHeadAttention<A, D> where D: Dimension, {

    pub fn head(&self) -> Option<&AttentionHead<A, D>> {
        self.attention.as_ref()
    }
    
    pub const fn config(&self) -> &Config {
        &self.config
    }

    pub fn linears(&self) -> &[Linear<A, Biased, D>] {
        &self.linears
    }
}

impl<A> MultiHeadAttention<A, Ix2> {
    pub fn std(config: Config) -> Self where A: Clone + Default {
        let linears = (0..4)
            .map(|_| Linear::from_features(config.d_model(), config.d_model()))
            .collect();
        Self {
            attention: None,
            config,
            linears
        }
    }
}