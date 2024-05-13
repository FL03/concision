/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::layout::{Features, Layout};
use crate::params::{Biased, Unbiased};
use core::marker::PhantomData;
use nd::prelude::*;
use nd::{IntoDimension, RemoveAxis, ShapeError};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Config<K = Biased, D = Ix2> {
    pub layout: Layout<D>,
    pub name: String,
    _biased: PhantomData<K>,
}

impl<K, D> Config<K, D>
where
    D: Dimension,
{
    pub fn new() -> Self {
        Self {
            layout: Layout::default(),
            name: String::new(),
            _biased: PhantomData::<K>,
        }
    }

    pub fn from_dim(dim: D) -> Result<Self, ShapeError>
    where
        D: Dimension,
    {
        let layout = Layout::from_dim(dim)?;
        let res = Self::new().with_layout(layout);
        Ok(res)
    }

    pub fn from_shape<Sh>(shape: Sh) -> Self
    where
        D: RemoveAxis,
        Sh: ShapeBuilder<Dim = D>,
    {
        let layout = Layout::from_shape(shape);
        Self::new().with_layout(layout)
    }

    pub fn into_biased(self) -> Config<Biased, D> {
        Config {
            layout: self.layout,
            name: self.name,
            _biased: PhantomData::<Biased>,
        }
    }

    pub fn into_unbiased(self) -> Config<Unbiased, D> {
        Config {
            layout: self.layout,
            name: self.name,
            _biased: PhantomData::<Unbiased>,
        }
    }

    pub fn with_name(self, name: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            ..self
        }
    }

    pub fn with_layout<E>(self, layout: Layout<E>) -> Config<K, E>
    where
        E: Dimension,
    {
        Config {
            layout,
            name: self.name,
            _biased: self._biased,
        }
    }
    /// Returns a cloned reference to the [dimension](ndarray::Dimension) of the [layout](Layout)
    pub fn dim(&self) -> D {
        self.layout().dim()
    }

    pub fn into_pattern(self) -> D::Pattern {
        self.dim().into_pattern()
    }
    /// This function attempts to convert the [layout](Layout) of the [Config] into a new [dimension](ndarray::Dimension)
    pub fn into_dimensionality<E>(self, dim: E) -> Result<Config<K, E>, nd::ShapeError>
    where
        E: Dimension,
    {
        let tmp = Config {
            layout: self.layout.into_dimensionality(dim)?,
            name: self.name,
            _biased: self._biased,
        };
        Ok(tmp)
    }
    /// Determine whether the [Config] is [Biased];
    /// Returns true by comparing the [TypeId](core::any::TypeId) of `K` against the [TypeId](core::any::TypeId) of the [Biased] type
    pub fn is_biased(&self) -> bool
    where
        K: 'static,
    {
        use core::any::TypeId;

        TypeId::of::<K>() == TypeId::of::<Biased>()
    }
    /// Returns an instance to the [Features] of the [Layout]
    pub fn features(&self) -> Features {
        self.layout().features()
    }
    /// Returns an owned reference to the [Layout]
    pub const fn layout(&self) -> &Layout<D> {
        &self.layout
    }
    /// Returns an immutable reference to the `name` of the model.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn ndim(&self) -> usize {
        self.layout().ndim()
    }
}

impl<K> Config<K, Ix2> {
    pub fn std(inputs: usize, outputs: usize) -> Self {
        Self {
            layout: Layout::new((outputs, inputs).into_dimension()),
            name: String::new(),
            _biased: PhantomData::<K>,
        }
    }
}

impl<D> Config<Biased, D>
where
    D: Dimension,
{
    /// The default constructor method for building [Biased] configurations.
    pub fn biased() -> Self {
        Self::new()
    }
}

impl<D> Config<Unbiased, D>
where
    D: Dimension,
{
    pub fn unbiased() -> Self {
        Self::new()
    }
}

impl<K, D> concision::Config for Config<K, D> where D: Dimension {}

impl<D> Default for Config<Biased, D>
where
    D: Dimension,
{
    fn default() -> Self {
        Self::new()
    }
}
