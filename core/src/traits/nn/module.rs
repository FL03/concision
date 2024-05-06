/*
   Appellation: modules <traits::nn>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

/// A `Module` defines any object that may be used as a layer in a neural network.
/// [Config](Module::Config) is a type that defines the configuration of the module; including any and all hyperparameters.
/// [Params](Module::Params) is a type that defines the parameters of the module; typically references a Linear set of parameters { weights, bias }
pub trait Module {
    type Config;
    type Params;

    fn config(&self) -> &Self::Config;

    fn params(&self) -> &Self::Params;

    fn params_mut(&mut self) -> &mut Self::Params;
}

pub trait NeuralNetwork: Module {
    
}