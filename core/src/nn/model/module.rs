/*
   Appellation: modules <traits::nn>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Config, Predict};

pub type ModuleDyn<C, P> = Box<dyn Module<Config = C, Params = P>>;
pub type DynModuleExt<X, Y, C, P> = Box<dyn ModuleExt<X, Config = C, Output = Y, Params = P>>;
pub type Stack<X, Y, C, P> = Vec<Box<dyn ModuleExt<X, Config = C, Output = Y, Params = P>>>;

/// A `Module` defines any object that may be used as a layer in a neural network.
/// [Config](Module::Config) is a type that defines the configuration of the module; including any and all hyperparameters.
/// [Params](Module::Params) is a type that defines the parameters of the module; typically references a Linear set of parameters { weights, bias }
pub trait Module {
    type Config: Config;
    type Params;

    fn config(&self) -> &Self::Config;

    fn params(&self) -> &Self::Params;

    fn params_mut(&mut self) -> &mut Self::Params;
}

pub trait ModuleExt<T>: Module + Predict<T> {}

impl<T, M> ModuleExt<T> for M where M: Module + Predict<T> {}
