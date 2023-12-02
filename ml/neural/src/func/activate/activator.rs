/*
    Appellation: activator <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{Activate, Gradient};
use ndarray::prelude::{Array, Dimension, Ix2};
use serde::{Deserialize, Serialize};

pub struct Activator<T = f64, D = Ix2>
where
    D: Dimension,
{
    method: Box<dyn Activate<T, D>>,
}

impl<T, D> Activator<T, D>
where
    D: Dimension,
{
    pub fn new(method: Box<dyn Activate<T, D>>) -> Self {
        Self { method }
    }

    pub fn from_method(method: impl Activate<T, D> + 'static) -> Self {
        Self::new(Box::new(method))
    }

    pub fn boxed(&self) -> &Box<dyn Activate<T, D>> {
        &self.method
    }

    pub fn method(&self) -> &dyn Activate<T, D> {
        self.method.as_ref()
    }
}

impl<T, D> Activator<T, D>
where
    D: Dimension,
    T: Clone,
{
    pub fn linear() -> Self {
        Self::new(Box::new(super::Linear::new()))
    }
}

impl<T, D> Activate<T, D> for Activator<T, D>
where
    D: Dimension,
    dyn Activate<T, D>: Activate<T, D>,
{
    fn activate(&self, args: &Array<T, D>) -> Array<T, D> {
        self.method.activate(args)
    }
}

impl<T, D> Gradient<T, D> for Activator<T, D>
where
    D: Dimension,
    dyn Activate<T, D>: Gradient<T, D>,
{
    fn gradient(&self, args: &Array<T, D>) -> Array<T, D> {
        self.method.gradient(args)
    }
}

impl<T, D> Clone for Activator<T, D>
where
    D: Dimension,
    T: Clone,
    dyn Activate<T, D>: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.method.clone())
    }
}

impl<T, D> Default for Activator<T, D>
where
    D: Dimension,
    dyn Activate<T, D>: Default,
{
    fn default() -> Self {
        Self::new(Box::new(<dyn Activate<T, D>>::default()))
    }
}

// impl<T, D> Copy for A<T, D> where D: Dimension, T: Copy, dyn Activate<T, D>: Copy {}

impl<T, D> PartialEq for Activator<T, D>
where
    D: Dimension,
    dyn Activate<T, D>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.method.eq(&other.method)
    }
}

impl<T, D> Eq for Activator<T, D>
where
    D: Dimension,
    dyn Activate<T, D>: Eq,
{
}

impl<T, D> std::fmt::Debug for Activator<T, D>
where
    D: Dimension,
    dyn Activate<T, D>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.method.fmt(f)
    }
}

impl<T, D> std::fmt::Display for Activator<T, D>
where
    D: Dimension,
    dyn Activate<T, D>: std::fmt::Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.method.fmt(f)
    }
}

impl<T, D> Serialize for Activator<T, D>
where
    D: Dimension,
    dyn Activate<T, D>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.method.serialize(serializer)
    }
}

impl<'de, T, D> Deserialize<'de> for Activator<T, D>
where
    D: Dimension,
    dyn Activate<T, D>: Deserialize<'de>,
{
    fn deserialize<D2>(deserializer: D2) -> Result<Self, D2::Error>
    where
        D2: serde::Deserializer<'de>,
    {
        Ok(Self::new(Box::new(<dyn Activate<T, D>>::deserialize(
            deserializer,
        )?)))
    }
}

impl<T, D> From<Box<dyn Activate<T, D>>> for Activator<T, D>
where
    D: Dimension,
{
    fn from(method: Box<dyn Activate<T, D>>) -> Self {
        Self::new(method)
    }
}
