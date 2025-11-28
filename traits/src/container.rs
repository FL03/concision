/*
    Appellation: container <module>
    Created At: 2025.11.28:15:30:46
    Contrib: @FL03
*/

pub trait Container {
    type Cont<U>;
    type Item;

}

impl<X> Container for Option<X> {
    type Cont<U> = Option<U>;
    type Item = X;
}