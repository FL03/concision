/*
    appellation: config <module>
    authors: @FL03
*/
//! This module is dedicated to establishing common interfaces for valid configuration objects
//! while providing a standard implementation to quickly spin up a new model.
#[doc(inline)]
pub use self::{model_config::StandardModelConfig, traits::*, types::*};

pub mod model_config;

mod traits {
    #[doc(inline)]
    pub use self::config::*;

    mod config;
}

mod types {
    #[doc(inline)]
    pub use self::{hyper_params::*, key_value::*};

    mod hyper_params;
    mod key_value;
}

pub(crate) mod prelude {
    pub use super::model_config::*;
    pub use super::traits::*;
    pub use super::types::*;
}

#[cfg(test)]
mod tests {
    use super::StandardModelConfig;

    #[test]
    fn test_standard_model_config() {
        // initialize a new model configuration with then given epochs and batch size
        let mut config = StandardModelConfig::new()
            .with_epochs(1000)
            .with_batch_size(32);
        // set various hyperparameters
        config.set_learning_rate(0.01);
        config.set_momentum(0.9);
        config.set_decay(0.0001);
        // verify the configuration
        assert_eq!(config.batch_size(), 32);
        assert_eq!(config.epochs(), 1000);
        // validate the stored hyperparameters
        assert_eq!(config.learning_rate(), Some(&0.01));
        assert_eq!(config.momentum(), Some(&0.9));
        assert_eq!(config.decay(), Some(&0.0001));
    }
}
