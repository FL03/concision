/*
   Appellation: model <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Backward, Forward};

pub trait Module {
    type Config;
    type Params;

    fn config(&self) -> Self::Config;

    fn params(&self) -> Self::Params;

    fn params_mut(&mut self) -> &mut Self::Params;
}

pub trait FeedForward<T>: Module
where
    Self: Backward + Forward<T>,
{
}

pub trait Model {
    type Backend: ModelBackend;
}

/// A trait for specifying the backend of a model.
///
/// The [Engine](ModelBackend::Engine) describes the type of nerual network being used; i.e. Convolution, Recurrant, Graph, etc.
pub trait ModelBackend {
    type Engine;
}
