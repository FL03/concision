/*
    Appellation: results <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub enum PropogationResult<T> {
    Forward(ForwardResult<T>),
    Backward(String),
}

pub enum ForwardResult<T> {
    Forward(T),
}
