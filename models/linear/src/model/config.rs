/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::layout::{Features, Layout};
use crate::params::{Biased, Unbiased};
use core::marker::PhantomData;
use nd::{Dimension, IntoDimension, Ix2, RemoveAxis};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Config<B = Biased, D = Ix2> {
    pub layout: Layout<D>,
    pub name: String,
    _biased: PhantomData<B>,
}

impl<K, D> Config<K, D>
where
    D: Dimension,
{
    pub fn new() -> Self {
        Self {
            layout: Layout::default(),
            name: String::new(),
            _biased: PhantomData,
        }
    }

    pub fn into_biased(self) -> Config<Biased, D> {
        Config {
            _biased: PhantomData,
            layout: self.layout,
            name: self.name,
        }
    }

    pub fn into_unbiased(self) -> Config<Unbiased, D> {
        Config {
            _biased: PhantomData,
            layout: self.layout,
            name: self.name,
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

    pub fn dim(&self) -> D {
        self.layout.dim().clone()
    }

    pub fn into_pattern(self) -> D::Pattern {
        self.dim().into_pattern()
    }

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

    pub fn is_biased(&self) -> bool
    where
        K: 'static,
    {
        use core::any::TypeId;

        TypeId::of::<K>() == TypeId::of::<Biased>()
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

impl<K> Config<K, Ix2> {
    pub fn std(inputs: usize, outputs: usize) -> Self {
        Self {
            layout: Layout::new((outputs, inputs).into_dimension()),
            name: String::new(),
            _biased: PhantomData,
        }
    }
}

impl<D> Config<Biased, D>
where
    D: Dimension,
{
    pub fn biased() -> Self {
        Self::new()
    }

    pub fn from_dim_biased(dim: D) -> Self
    where
        D: RemoveAxis,
    {
        let layout = Layout::from_dim(dim).unwrap();
        Self::new().with_layout(layout)
    }
}

impl<D> Config<Unbiased, D>
where
    D: Dimension,
{
    pub fn unbiased() -> Self {
        Self::new()
    }

    pub fn from_dim(dim: D) -> Config<Unbiased, D>
    where
        D: RemoveAxis,
    {
        Config::<Unbiased, D>::new().with_layout(Layout::from_dim(dim).unwrap())
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
