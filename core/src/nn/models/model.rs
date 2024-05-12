/*
    Appellation: model <traits>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::{DynModule, Module};
use crate::traits::Forward;

pub trait Model: Module
where
    Self: Forward<Self::Data>,
{
    type Ctx;
    type Data;

    fn children(&self) -> Vec<DynModule<Self::Ctx, Self::Params>>;
}

pub struct ConfigBase {
    pub id: usize,
    pub name: String,
}
