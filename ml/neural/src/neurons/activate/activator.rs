/*
    Appellation: activator <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Activate, ActivationMethod, ActivationFn};
use std::marker::PhantomData;

pub trait ActivationParams {}

pub trait LinearActivation<T>: ActivationMethod {
    fn rho() -> ActivationFn<T> {
        |x| x
    }

    fn linear(&self, x: T) -> T {
        x
    }
}

pub struct Activator<T, A> where A: Activate<T> {
    method: A,
    _args: PhantomData<T>
}

impl<T, A> Activator<T, A> where A: Activate<T> {
    pub fn new(method: A) -> Self {
        Activator {
            method,
            _args: PhantomData
        }
    }
}

impl<T, A> Activate<T> for Activator<T, A> where A: Activate<T> {
    fn activate(&self, x: T) -> T {
        self.method.activate(x)
    }
}




