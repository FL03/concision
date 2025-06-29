/*
    Appellation: config <module>
    Contrib: @FL03
*/

/// The [`RawConfig`] trait defines a basic interface for all _configurations_ used within the
/// framework for neural networks, their layers, and more.
pub trait RawConfig: Send + Sync + core::fmt::Debug {
    type Ctx;
}

/// The [`NetworkConfig`] trait extends the [`RawConfig`] trait to provide a more robust
/// interface for neural network configurations.
pub trait NetworkConfig<T>: RawConfig<Ctx = T> {
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

    fn keys(&self) -> Vec<String>;
}

macro_rules! hyperparam_method {
    (@dyn $name:ident: $type:ty) => {
        fn $name(&self) -> Option<&$type> where T: 'static {
            self.get(stringify!($name)).map(|v| v.downcast_ref::<$type>()).flatten()
        }
    };
    (@impl $name:ident: $type:ty) => {
        fn $name(&self) -> Option<&$type> {
            self.get(stringify!($name))
        }
    };
    (#[dyn] $($name:ident $type:ty),* $(,)?) => {
        $(
            hyperparam_method!(@dyn $name: $type);
        )*
    };
    ($($name:ident $type:ty),* $(,)?) => {
        $(
            hyperparam_method!(@impl $name: $type);
        )*
    };
}

pub trait TrainingConfiguration<T>: NetworkConfig<T> {
    fn epochs(&self) -> usize;

    fn batch_size(&self) -> usize;

    hyperparam_method! {
        learning_rate T,
        momentum T,
        weight_decay T,
        dropout T,
        decay T,
        beta1 T,
        beta2 T,
        epsilon T,
        gradient_clip T,

    }
}

/*
 ************* Implementations *************
*/
