/*
    Appellation: container <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Container<T> {
    type Data: Data<Item = T>;
}

pub trait Data {
    type Item;
}
