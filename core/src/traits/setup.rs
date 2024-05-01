/*
   Appellation: setup <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Initialize {
    fn init(&mut self);
}

pub trait Setup {
    type Config;

    fn setup(&mut self, config: Self::Config);
}
