/*
   Appellation: block <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::func::activate::{Activate, LinearActivation, ReLU, Softmax};
use num::Float;
use std::marker::PhantomData;

pub trait FnBlock<T> {
    fn apply(&self, data: T) -> T;
}

#[derive(Clone)]
pub struct FuncBlock<T = f64> {
    method: Vec<fn(&T) -> T>,
}

pub struct FFNBlock<T = f64, I = LinearActivation, H = ReLU, O = Softmax>
where
    I: Activate<T>,
    H: Activate<T>,
    O: Activate<T>,
    T: Float,
{
    _args: PhantomData<T>,
    input: I,
    hidden: H,
    output: O,
}

#[cfg(test)]
mod tests {}
