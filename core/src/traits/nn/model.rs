/*
   Appellation: model <traits>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Model {
    type Backend: ModelBackend;
}

/// A trait for specifying the backend of a model.
///
/// The [Engine](ModelBackend::Engine) describes the type of nerual network being used; i.e. Convolution, Recurrant, Graph, etc.
pub trait ModelBackend {
    type Engine;
}

pub trait NeuralNetworkStack {
    const NHIDDEN: Option<usize> = None;
    type Input;
    type Hidden;
    type Output;
}

#[allow(dead_code)]
pub struct ModelBase<C, P> {
    pub(crate) id: usize,
    config: C,
    params: P,
}
