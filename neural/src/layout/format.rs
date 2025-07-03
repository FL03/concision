/*
    appellation: format <module>
    authors: @FL03
*/

/// The [`ModelFormat`] type enumerates the various formats a neural network may take, either
/// shallow or deep, providing a unified interface for accessing the number of hidden features
/// and layers in the model. This is done largely for simplicity, as it eliminates the need to
/// define a particular _type_ of network as its composition has little impact on the actual
/// requirements / algorithms used to train or evaluate the model (that is, outside of the
/// obvious need to account for additional hidden layers in deep configurations). In other
/// words, both shallow and deep networks are requried to implement the same traits and
/// fulfill the same requirements, so it makes sense to treat them as a single type with
/// different configurations. The differences between the networks are largely left to the
/// developer and their choice of activation functions, optimizers, and other considerations.
#[derive(
    Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, strum::EnumCount, strum::EnumIs,
)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum ModelFormat {
    Shallow { hidden: usize },
    Deep { hidden: usize, layers: usize },
}

impl ModelFormat {
    /// initialize a new [`Deep`](ModelFormat::Deep) variant for a deep neural network with the
    /// given number of hidden features and layers
    pub const fn deep(hidden: usize, layers: usize) -> Self {
        ModelFormat::Deep { hidden, layers }
    }
    /// create a new instance of [`ModelFormat`] for a shallow neural network, using the given
    /// number of hidden features
    pub const fn shallow(hidden: usize) -> Self {
        ModelFormat::Shallow { hidden }
    }
    /// returns a copy of the number of hidden features
    pub const fn hidden(&self) -> usize {
        match self {
            ModelFormat::Shallow { hidden } => *hidden,
            ModelFormat::Deep { hidden, .. } => *hidden,
        }
    }
    /// returns a mutable reference to the hidden features for the model
    pub const fn hidden_mut(&mut self) -> &mut usize {
        match self {
            ModelFormat::Shallow { hidden } => hidden,
            ModelFormat::Deep { hidden, .. } => hidden,
        }
    }
    /// returns a copy of the number of layers for the model; if the variant is
    /// [`Shallow`](ModelFormat::Shallow), it returns 1
    /// returns `n` if the variant is [`Deep`](ModelFormat::Deep)
    pub const fn layers(&self) -> usize {
        match self {
            ModelFormat::Shallow { .. } => 1,
            ModelFormat::Deep { layers, .. } => *layers,
        }
    }
    /// returns a mutable reference to the number of layers for the model; this will panic on
    /// [`Shallow`](ModelFormat::Shallow) variants
    pub const fn layers_mut(&mut self) -> &mut usize {
        match self {
            ModelFormat::Shallow { .. } => panic!("Cannot mutate layers of a shallow model"),
            ModelFormat::Deep { layers, .. } => layers,
        }
    }
    /// update the number of hidden features for the model
    pub fn set_hidden(&mut self, value: usize) -> &mut Self {
        match self {
            ModelFormat::Shallow { hidden } => {
                *hidden = value;
            }
            ModelFormat::Deep { hidden, .. } => {
                *hidden = value;
            }
        }
        self
    }
    /// update the number of layers for the model;
    ///
    /// **note:** this method will automatically convert the model to a [`Deep`](ModelFormat::Deep)
    /// variant if it is currently a [`Shallow`](ModelFormat::Shallow) variant and the number
    /// of layers becomes greater than 1
    pub fn set_layers(&mut self, value: usize) -> &mut Self {
        match self {
            ModelFormat::Shallow { hidden } => {
                if value > 1 {
                    *self = ModelFormat::Deep {
                        hidden: *hidden,
                        layers: value,
                    };
                }
                // if the value is 1, we do not change the model format
            }
            ModelFormat::Deep { layers, .. } => {
                *layers = value;
            }
        }
        self
    }
    /// consumes the current instance and returns a new instance with the given hidden
    /// features
    pub fn with_hidden(self, hidden: usize) -> Self {
        match self {
            ModelFormat::Shallow { .. } => ModelFormat::Shallow { hidden },
            ModelFormat::Deep { layers, .. } => ModelFormat::Deep { hidden, layers },
        }
    }
    /// consumes the current instance and returns a new instance with the given number of
    /// hidden layers
    ///
    /// **note:** this method will automatically convert the model to a [`Deep`](ModelFormat::Deep)
    /// variant if it is currently a [`Shallow`](ModelFormat::Shallow) variant and the number
    /// of layers becomes greater than 1
    pub fn with_layers(self, layers: usize) -> Self {
        match self {
            ModelFormat::Shallow { hidden } => {
                if layers > 1 {
                    ModelFormat::Deep { hidden, layers }
                } else {
                    ModelFormat::Shallow { hidden }
                }
            }
            ModelFormat::Deep { hidden, .. } => ModelFormat::Deep { hidden, layers },
        }
    }
}

impl Default for ModelFormat {
    fn default() -> Self {
        Self::Deep {
            hidden: 16,
            layers: 1,
        }
    }
}

impl core::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{{ hidden: {}, layers: {} }}",
            self.hidden(),
            self.layers()
        )
    }
}
