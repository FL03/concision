/*
    Appellation: config <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::layout::{Features, Layout};
use super::{Biased, ParamMode, Unbiased};
use core::marker::PhantomData;
use nd::{Dimension, IntoDimension, Ix2, RemoveAxis};

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Config<D = Ix2, B = Biased>
where
    D: Dimension,
{
    pub layout: Layout<D>,
    pub name: String,
    _biased: PhantomData<B>,
}

impl<D, K> Config<D, K>
where
    D: Dimension,
    K: ParamMode,
{
    pub fn new(layout: Layout<D>, name: impl ToString) -> Self {
        Self {
            layout,
            name: name.to_string(),
            _biased: PhantomData,
        }
    }

    pub fn from_dim(dim: D) -> Self
    where
        D: RemoveAxis,
    {
        Self::new(Layout::from_dim(dim).unwrap(), "")
    }

    pub fn biased(self) -> Config<D, Biased> {
        Config {
            _biased: PhantomData,
            layout: self.layout,
            name: self.name,
        }
    }

    pub fn unbiased(self) -> Config<D, Unbiased> {
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

    pub fn with_layout<E>(self, layout: Layout<E>) -> Config<E, K>
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

    pub fn into_dimensionality<E>(self, dim: E) -> Result<Config<E, K>, nd::ShapeError>
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

impl<K> Config<Ix2, K> {
    pub fn std(inputs: usize, outputs: usize) -> Self {
        Self {
            layout: Layout::new((outputs, inputs).into_dimension()),
            name: String::new(),
            _biased: PhantomData,
        }
    }
}

impl<D> Config<D, Biased> where D: Dimension {}

impl<D> Config<D, Unbiased> where D: Dimension {}
