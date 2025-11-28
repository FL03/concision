/*
    appellation: config <module>
    authors: @FL03
*/
#[doc(inline)]
pub use self::{model_config::StandardModelConfig, traits::*, types::*};

pub mod model_config;

mod traits {
    #[doc(inline)]
    pub use self::prelude::*;

    mod config;

    mod prelude {
        #[doc(inline)]
        pub use super::config::*;
    }
}

mod types {
    //! this module defines various types in-support of the configuration model for the neural
    //! library of the concision framework.
    #[doc(inline)]
    pub use self::prelude::*;

    mod hyper_params;

    mod prelude {
        #[doc(inline)]
        pub use super::hyper_params::*;
    }
}

pub(crate) mod prelude {
    #[doc(inline)]
    pub use super::model_config::*;
    #[doc(inline)]
    pub use super::traits::*;
    #[doc(inline)]
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
