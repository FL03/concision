/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Features;
use crate::model::Layout;
use nd::{Dimension, Ix2, RemoveAxis};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Config<D = Ix2>
where
    D: Dimension,
{
    pub biased: bool,
    pub layout: Layout<D>,
    pub name: String,
}

impl<D> Config<D>
where
    D: Dimension,
{
    pub fn from_dim(dim: D) -> Self
    where
        D: RemoveAxis,
    {
        Self {
            biased: false,
            layout: Layout::new(dim),
            name: String::new(),
        }
    }

    pub fn is_biased(&self) -> bool {
        self.biased
    }

    pub fn biased(self) -> Self {
        Self {
            biased: true,
            ..self
        }
    }

    pub fn unbiased(self) -> Self {
        Self {
            biased: false,
            ..self
        }
    }

    pub fn with_name(self, name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            ..self
        }
    }

    pub fn with_layout<E>(self, layout: Layout<E>) -> Config<E>
    where
        E: Dimension,
    {
        Config {
            biased: self.biased,
            layout,
            name: self.name,
        }
    }

    pub fn into_pattern(self) -> D::Pattern {
        self.dim().into_pattern()
    }

    pub fn into_dimensionality<E>(self, dim: E) -> Result<Config<E>, nd::ShapeError>
    where
        E: Dimension,
    {
        let tmp = Config {
            biased: self.biased,
            layout: self.layout.into_dimensionality(dim)?,
            name: self.name,
        };
        Ok(tmp)
    }

    pub fn dim(&self) -> D {
        self.layout.dim().clone()
    }

    pub fn features(&self) -> Features {
        self.layout.features()
    }

    pub fn layout(&self) -> &Layout<D> {
        &self.layout
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ndim(&self) -> usize {
        self.layout.ndim()
    }
}
