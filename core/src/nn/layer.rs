/*
    Appellation: layer <module>
    Contrib: @FL03
*/
use crate::Params;

pub struct Layer<A> {
    pub activation: Box<dyn Fn(A) -> A>,
    pub params: Params<A>,
}

impl<A> Layer<A>
where
    A: 'static,
{
    pub fn new<F>(activation: F, params: Params<A>) -> Self
    where
        F: Fn(A) -> A + 'static,
    {
        Self {
            activation: Box::new(activation),
            params,
        }
    }

    pub fn sigmoid(params: Params<A>) -> Self
    where
        A: num_traits::Float,
    {
        Self::new(crate::activate::sigmoid, params)
    }
}
