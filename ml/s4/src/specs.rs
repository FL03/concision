/*
    Appellation: specs <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Scan<S, T> {
    type Output;

    fn scan(&self, args: &T, initial_state: &S) -> Self::Output;
}

pub trait StateSpace<T> {
    type Config;

    fn config(&self) -> &Self::Config;
}
