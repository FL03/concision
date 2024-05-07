/*
    Appellation: config <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::logstep;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub struct SSMConfig {
    pub decode: bool,
    pub features: usize, // n
    pub samples: usize,  // l_max
}

impl SSMConfig {
    pub fn new(decode: bool, features: usize, samples: usize) -> Self {
        Self {
            decode,
            features,
            samples,
        }
    }

    pub fn decode(&self) -> bool {
        self.decode
    }

    pub fn features(&self) -> usize {
        self.features
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    pub fn step_size<T>(&self) -> T
    where
        T: num::Float,
    {
        T::from(self.samples).unwrap().recip()
    }

    pub fn logstep(&self) -> f64 {
        logstep::<f64>(1e-3, 1e-1).exp()
    }

    pub fn set_decode(&mut self, decode: bool) {
        self.decode = decode;
    }

    pub fn set_features(&mut self, features: usize) {
        self.features = features;
    }

    pub fn set_samples(&mut self, samples: usize) {
        self.samples = samples;
    }

    pub fn with_decode(mut self, decode: bool) -> Self {
        self.decode = decode;
        self
    }

    pub fn with_features(mut self, features: usize) -> Self {
        self.features = features;
        self
    }

    pub fn with_samples(mut self, samples: usize) -> Self {
        self.samples = samples;
        self
    }
}
