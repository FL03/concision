/*
   Appellation: base <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Walk<T> {
    fn walk(&self, other: &T) -> bool;
}
