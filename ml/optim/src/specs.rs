/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Optimize {
    fn optimize(&self, params: &mut dyn Optimizable);
}

pub trait Optimizable {}
