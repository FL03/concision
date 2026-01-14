/*
    appellation: layout <module>
    authors: @FL03
*/
use super::{Deep, NetworkDepth, RawModelLayout};

mod impl_model_features;
mod impl_model_format;
mod impl_model_layout;

/// A trait that consumes the caller to create a new instance of [`ModelFeatures`] object.
pub trait IntoModelFeatures {
    fn into_model_features(self) -> ModelFeatures;
}

/// The [`ModelFormat`] type enumerates the various formats a neural network may take, either
/// shallow or deep, providing a unified interface for accessing the number of hidden features
/// and layers in the model. This is primarily used to generalize the allowed formats of a
/// neural network without introducing any additional complexity with typing or other
/// constructs.
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, strum::EnumCount, strum::EnumIs,
)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ModelFormat {
    Layer,
    Shallow { hidden: usize },
    Deep { hidden: usize, layers: usize },
}

/// The [`ModelFeatures`] provides a common way of defining the layout of a model. This is
/// used to define the number of input features, the number of hidden layers, the number of
/// hidden features, and the number of output features.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ModelFeatures {
    /// the number of input features
    pub(crate) input: usize,
    /// the features of the "inner" layers
    pub(crate) inner: ModelFormat,
    /// the number of output features
    pub(crate) output: usize,
}

/// In contrast to the [`ModelFeatures`] type, the [`ModelLayout`] implementation aims to
/// provide a generic foundation for using type-based features / layouts within neural network.
/// Our goal with this struct is to eventually push the implementation to the point of being
/// able to sufficiently describe everything about a model's layout (similar to what the
/// [`ndarray`] developers have attained with the [`LayoutRef`](ndarray::LayoutRef)).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ModelLayout<F, D = Deep>
where
    D: NetworkDepth,
    F: RawModelLayout,
{
    pub(crate) features: F,
    pub(crate) _marker: core::marker::PhantomData<D>,
}

/*
 ************* Implementations *************
*/

impl IntoModelFeatures for (usize, usize, usize) {
    fn into_model_features(self) -> ModelFeatures {
        ModelFeatures {
            input: self.0,
            inner: ModelFormat::Shallow { hidden: self.1 },
            output: self.2,
        }
    }
}

impl IntoModelFeatures for (usize, usize, usize, usize) {
    fn into_model_features(self) -> ModelFeatures {
        ModelFeatures {
            input: self.0,
            inner: ModelFormat::Deep {
                hidden: self.1,
                layers: self.3,
            },
            output: self.2,
        }
    }
}

impl IntoModelFeatures for [usize; 3] {
    fn into_model_features(self) -> ModelFeatures {
        ModelFeatures {
            input: self[0],
            inner: ModelFormat::Shallow { hidden: self[1] },
            output: self[2],
        }
    }
}

impl IntoModelFeatures for [usize; 4] {
    fn into_model_features(self) -> ModelFeatures {
        ModelFeatures {
            input: self[0],
            inner: ModelFormat::Deep {
                hidden: self[1],
                layers: self[3],
            },
            output: self[2],
        }
    }
}
