/*
    Appellation: activator <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Activate;

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

impl<T> Activate<T> for Activator<T> {
    fn activate(&self, x: T) -> T {
        self.method().activate(x)
    }
}
