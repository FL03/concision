/*
    Appellation: config <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::Features;

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Config {
    pub biased: bool,
    pub name: String,
    pub shape: Features,
}

impl Config {
    pub fn new(name: impl ToString, shape: Features) -> Self {
        Self {
            biased: false,
            name: name.to_string(),
            shape,
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

    pub fn with_shape(self, shape: Features) -> Self {
        Self { shape, ..self }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            biased: false,
            name: String::new(),
            shape: Features::default(),
        }
    }
}
