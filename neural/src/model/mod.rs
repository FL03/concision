/*
    Appellation: model <module>
    Contrib: @FL03
*/
//! This module provides the scaffolding for creating models and layers in a neural network.

pub trait Layer {
    type Input;
    type Output;

    fn forward(&self, input: &Self::Input) -> cnc::CncResult<Self::Output>;
    fn backward(&self, input: &Self::Input) -> cnc::CncResult<Self::Output>;
}

pub trait Model {}
