/*
    Appellation: features <module>
    Contrib: @FL03
*/

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ModelFeatures {
    /// the number of input features
    pub(crate) input: usize,
    /// the dimension of hidden layers
    pub(crate) hidden: usize,
    /// the number of hidden layers
    pub(crate) layers: usize,
    /// the number of output features
    pub(crate) output: usize,
}

impl ModelFeatures {
    pub fn new(input: usize, hidden: usize, layers: usize, output: usize) -> Self {
        Self {
            input,
            hidden,
            layers,
            output,
        }
    }

    gsw! {
        input: usize,
        hidden: usize,
        layers: usize,
        output: usize,
    }
    /// the dimension of the input layer; (input, hidden)
    pub fn d_input(&self) -> (usize, usize) {
        (self.input, self.hidden)
    }
    /// the dimension of the hidden layers; (hidden, hidden)
    pub fn d_hidden(&self) -> (usize, usize) {
        (self.hidden, self.hidden)
    }
    /// the dimension of the output layer; (hidden, output)
    pub fn d_output(&self) -> (usize, usize) {
        (self.hidden, self.output)
    }
    /// the total number of parameters in the model
    pub fn size(&self) -> usize {
        self.input * self.hidden
            + self.hidden * self.hidden * self.layers
            + self.hidden * self.output
    }
    /// the total number of input parameters in the model
    pub fn size_input(&self) -> usize {
        self.input * self.hidden
    }
    /// the total number of hidden parameters in the model
    pub fn size_hidden(&self) -> usize {
        self.hidden * self.hidden * self.layers
    }
    /// the total number of output parameters in the model
    pub fn size_output(&self) -> usize {
        self.hidden * self.output
    }
}

impl Default for ModelFeatures {
    fn default() -> Self {
        Self {
            input: 16,
            hidden: 64,
            layers: 3,
            output: 16,
        }
    }
}

impl core::fmt::Display for ModelFeatures {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "ModelFeatures {{ input: {}, hidden: {}, layers: {}, output: {} }}",
            self.input, self.hidden, self.layers, self.output
        )
    }
}
