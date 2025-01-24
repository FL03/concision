/*
   Appellation: setup <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait FromConfig<Cnf> {
    fn from_config(config: Cnf) -> Self;
}

/// A trait used to denote objects that may be used for configuring various items
pub trait Config {}

pub trait CompositeConfig: Config {
    type Ctx: Config;

    fn children(&self) -> Vec<Box<dyn Config>>;

    fn context(&self) -> Self::Ctx;
}

pub trait Configurable {
    type Config: Config;

    fn configure(config: Self::Config) -> Self;
}

/*
 ************* Implementations *************
*/
