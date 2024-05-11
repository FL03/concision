/*
    Appellation: sequential <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::Forward;
use serde::{Deserialize, Serialize};

pub struct Sequential<T> {
    layers: Vec<Box<dyn Forward<T, Output = T>>>,
}

impl<T> Sequential<T> {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn include<L>(mut self, layer: L) -> Self
    where
        L: Forward<T, Output = T> + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }

    pub fn push<L>(&mut self, layer: L)
    where
        L: Forward<T, Output = T> + 'static,
    {
        self.layers.push(Box::new(layer));
    }
}

impl<T> AsRef<[Box<dyn Forward<T, Output = T>>]> for Sequential<T> {
    fn as_ref(&self) -> &[Box<dyn Forward<T, Output = T>>] {
        &self.layers
    }
}

impl<T> AsMut<[Box<dyn Forward<T, Output = T>>]> for Sequential<T> {
    fn as_mut(&mut self) -> &mut [Box<dyn Forward<T, Output = T>>] {
        &mut self.layers
    }
}

impl<T> Extend<Box<dyn Forward<T, Output = T>>> for Sequential<T> {
    fn extend<I: IntoIterator<Item = Box<dyn Forward<T, Output = T>>>>(&mut self, iter: I) {
        self.layers.extend(iter);
    }
}

impl<T> Forward<T> for Sequential<T>
where
    T: Clone,
{
    type Output = T;

    fn forward(&self, input: &T) -> Self::Output {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output);
        }
        output
    }
}

impl<T> FromIterator<Box<dyn Forward<T, Output = T>>> for Sequential<T> {
    fn from_iter<I: IntoIterator<Item = Box<dyn Forward<T, Output = T>>>>(iter: I) -> Self {
        Self {
            layers: Vec::from_iter(iter),
        }
    }
}

impl<T> IntoIterator for Sequential<T> {
    type Item = Box<dyn Forward<T, Output = T>>;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.into_iter()
    }
}

impl<T> Clone for Sequential<T>
where
    Box<dyn Forward<T, Output = T>>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            layers: self.layers.clone(),
        }
    }
}

impl<T> Default for Sequential<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> std::fmt::Debug for Sequential<T>
where
    Box<dyn Forward<T, Output = T>>: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("layers", &self.layers)
            .finish()
    }
}

impl<T> PartialEq for Sequential<T>
where
    Box<dyn Forward<T, Output = T>>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.layers == other.layers
    }
}

impl<T> Eq for Sequential<T> where Box<dyn Forward<T, Output = T>>: Eq {}

impl<T> std::hash::Hash for Sequential<T>
where
    Box<dyn Forward<T, Output = T>>: std::hash::Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.layers.hash(state);
    }
}

impl<T> PartialOrd for Sequential<T>
where
    Box<dyn Forward<T, Output = T>>: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.layers.partial_cmp(&other.layers)
    }
}

impl<T> Ord for Sequential<T>
where
    Box<dyn Forward<T, Output = T>>: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.layers.cmp(&other.layers)
    }
}

impl<'a, T> Deserialize<'a> for Sequential<T>
where
    Box<dyn Forward<T, Output = T>>: Deserialize<'a>,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'a>,
    {
        let layers = Vec::<Box<dyn Forward<T, Output = T>>>::deserialize(deserializer)?;
        Ok(Self { layers })
    }
}

impl<T> Serialize for Sequential<T>
where
    Box<dyn Forward<T, Output = T>>: Serialize,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.layers.serialize(serializer)
    }
}
