/*
   Appellation: flow <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Flow<T> {
    fn flow(&self, input: T) -> T;
}
