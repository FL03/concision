/*
    Appellation: activator <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Activate, LinearActivation};
use std::marker::PhantomData;

pub struct Activator<T = f64, A = LinearActivation>
where
    A: Activate<T>,
{
    method: A,
    _args: PhantomData<T>,
}

impl<T, A> Activator<T, A>
where
    A: Activate<T>,
{
    pub fn new(method: A) -> Self {
        Activator {
            method,
            _args: PhantomData,
        }
    }
}

impl<T, A> Activate<T> for Activator<T, A>
where
    A: Activate<T>,
{
    fn activate(&self, x: T) -> T {
        self.method.activate(x)
    }
}

