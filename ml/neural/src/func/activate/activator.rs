/*
    Appellation: activator <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::ActivateMethod;

pub struct Activator<T = f64> {
    method: Box<dyn ActivateMethod<T>>,
}

impl<T> Activator<T> {
    pub fn new(method: Box<dyn ActivateMethod<T>>) -> Self {
        Self { method }
    }

    pub fn method(&self) -> &dyn ActivateMethod<T> {
        self.method.as_ref()
    }
}

impl<T> ActivateMethod<T> for Activator<T> {
    fn rho(&self, x: T) -> T {
        self.method().rho(x)
    }
}
