/*
   Appellation: init <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Init {
    fn init(&mut self) -> Self;
}

pub trait InitRandom<T> {
    fn genrand(&mut self) -> T;
}

pub trait Rand {}

pub trait RandComplex {}
