/*
   Appellation: rank <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/


pub enum Rank<T> {
    Zero(T),
    One(Vec<T>),
    N(Vec<Self>),
}
