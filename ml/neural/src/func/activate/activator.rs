/*
    Appellation: activator <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Activate;

pub trait Activations<T = f64> {
    fn linear() -> Activator<T>
    where
        Self: Sized;
}

pub struct A<T = f64>(Box<dyn Activations<T>>);

pub struct Activator<T = f64> {
    method: Box<dyn Activate<T>>,
}

impl<T> Activator<T> {
    pub fn new(method: Box<dyn Activate<T>>) -> Self {
        Self { method }
    }

    pub fn method(&self) -> &dyn Activate<T> {
        self.method.as_ref()
    }
}
