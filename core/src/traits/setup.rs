/*
   Appellation: setup <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use core::borrow::{Borrow, BorrowMut};

/// A trait used to denote objects that may be used for configuring various items
pub trait Config {}

pub trait CompositConfig: Config {
    type Ctx: Config;

    fn children(&self) -> Vec<Box<dyn Config>>;

    fn context(&self) -> Self::Ctx;
}

/// [Configuration] describes composite configuration objects;
/// A configuration object is allowed to inherit from another configuration object
pub trait Configuration<C>
where
    C: Config,
    Self::Config: Borrow<C>,
{
    type Config: Config;

    fn root(&self) -> &C;

    fn set(&mut self, config: Self::Config);

    fn set_root(&mut self, config: C);
}

pub trait Init {
    fn init(self) -> Self;
}

pub trait InitInplace {
    fn init(&mut self);
}

pub trait Setup {
    type Config;

    fn setup(&mut self, config: Self::Config);
}

pub trait Context<C>
where
    C: Config,
{
    type Cnf: Configuration<C>;

    fn config(&self) -> Self::Cnf;
}

/*
 ************* Implementations *************
*/

impl<C, D> Configuration<C> for D
where
    C: Config,
    D: Config + BorrowMut<C>,
{
    type Config = D;

    fn root(&self) -> &C {
        self.borrow()
    }

    fn set(&mut self, config: Self::Config) {
        *self = config;
    }

    fn set_root(&mut self, config: C) {
        *self.borrow_mut() = config;
    }
}
