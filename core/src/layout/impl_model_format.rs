/*
    appellation: format <module>
    authors: @FL03
*/
use super::ModelFormat;

impl ModelFormat {
    pub const fn new(hidden: usize, layers: usize) -> Self {
        match layers {
            0 | 1 => ModelFormat::Shallow { hidden },
            _ => ModelFormat::Deep { hidden, layers },
        }
    }

    pub const fn layout() -> Self {
        ModelFormat::Layer
    }
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
            ModelFormat::Layer => 0,
        }
    }
    /// returns a mutable reference to the hidden features for the model
    pub const fn hidden_mut(&mut self) -> &mut usize {
        match self {
            ModelFormat::Shallow { hidden } => hidden,
            ModelFormat::Deep { hidden, .. } => hidden,
            ModelFormat::Layer => panic!("Cannot mutate hidden features of a layout model"),
        }
    }
    /// returns a copy of the number of layers for the model; if the variant is
    /// [`Shallow`](ModelFormat::Shallow), it returns 1
    /// returns `n` if the variant is [`Deep`](ModelFormat::Deep)
    pub const fn layers(&self) -> usize {
        match self {
            ModelFormat::Shallow { .. } => 1,
            ModelFormat::Deep { layers, .. } => *layers,
            ModelFormat::Layer => 0,
        }
    }
    /// returns a mutable reference to the number of layers for the model; this will panic on
    /// [`Shallow`](ModelFormat::Shallow) variants
    pub const fn layers_mut(&mut self) -> &mut usize {
        match self {
            ModelFormat::Shallow { .. } => panic!("Cannot mutate layers of a shallow model"),
            ModelFormat::Deep { layers, .. } => layers,
            ModelFormat::Layer => panic!("Cannot mutate layers of a layout model"),
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
            ModelFormat::Layer => {
                panic!("Cannot mutate hidden features of a layout model");
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
            ModelFormat::Layer => {
                panic!("Cannot mutate layers of a layout model");
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
            ModelFormat::Layer => ModelFormat::Shallow { hidden },
        }
    }
    /// consumes the current instance and returns a new instance with the given number of
    /// hidden layers
    ///
    /// **note:** this method will automatically convert the model to a [`Deep`](ModelFormat::Deep)
    /// variant if it is currently a [`Shallow`](ModelFormat::Shallow) variant and the number
    /// of layers becomes greater than 1
    pub fn with_layers(self, layers: usize) -> Self {
        match layers {
            0 => ModelFormat::Layer,
            1 => match self {
                ModelFormat::Shallow { hidden } => ModelFormat::Shallow { hidden },
                ModelFormat::Deep { hidden, .. } => ModelFormat::Shallow { hidden },
                ModelFormat::Layer => ModelFormat::Layer,
            },
            _ => match self {
                ModelFormat::Shallow { hidden } => ModelFormat::Deep { hidden, layers },
                ModelFormat::Deep { hidden, .. } => ModelFormat::Deep { hidden, layers },
                ModelFormat::Layer => ModelFormat::Deep { hidden: 16, layers },
            },
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
