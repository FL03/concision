/*
    appellation: config <module>
    authors: @FL03
*/
//! This module is dedicated to establishing common interfaces for valid configuration objects
//! while providing a standard implementation to quickly spin up a new model.
#[doc(inline)]
pub use self::{hyper_params::HyperParam, model_config::StandardModelConfig};

pub mod hyper_params;
pub mod model_config;
// prelude (local)
#[doc(hidden)]
pub(crate) mod prelude {
    pub use super::hyper_params::HyperParam;
    pub use super::model_config::*;
    pub use super::{ExtendedModelConfig, ModelConfiguration, RawConfig};
}

/// The [`RawConfig`] trait defines a basic interface for all _configurations_ used within the
/// framework for neural networks, their layers, and more.
pub trait RawConfig {
    type Ctx;
}

/// The [`ModelConfiguration`] trait extends the [`RawConfig`] trait to provide a more robust
/// interface for neural network configurations.
pub trait ModelConfiguration<T>: RawConfig {
    fn get<K>(&self, key: K) -> Option<&T>
    where
        K: AsRef<str>;
    fn get_mut<K>(&mut self, key: K) -> Option<&mut T>
    where
        K: AsRef<str>;

    fn set<K>(&mut self, key: K, value: T) -> Option<T>
    where
        K: AsRef<str>;
    fn remove<K>(&mut self, key: K) -> Option<T>
    where
        K: AsRef<str>;
    fn contains<K>(&self, key: K) -> bool
    where
        K: AsRef<str>;

    fn keys(&self) -> Vec<&str>;
}

macro_rules! hyperparam_method {
    ($($(dyn)? $name:ident::<$type:ty>),* $(,)?) => {
        $(
            hyperparam_method!(@impl $name::<$type>);
        )*
    };
    (@impl dyn $name:ident::<$type:ty>) => {
        fn $name(&self) -> Option<&$type> where T: 'static {
            self.get(stringify!($name)).map(|v| v.downcast_ref::<$type>()).flatten()
        }
    };
    (@impl $name:ident::<$type:ty>) => {
        fn $name(&self) -> Option<&$type> {
            self.get(stringify!($name))
        }
    };
}

pub trait ExtendedModelConfig<T>: ModelConfiguration<T> {
    fn epochs(&self) -> usize;

    fn batch_size(&self) -> usize;

    hyperparam_method! {
        learning_rate::<T>,
        epsilon::<T>,
        momentum::<T>,
        weight_decay::<T>,
        dropout::<T>,
        decay::<T>,
        beta::<T>,
        beta1::<T>,
        beta2::<T>,
    }
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
