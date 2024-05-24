/*
    Appellation: data <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait ContainerRepr {
    type Elem;
}

/*
 ************* Implementations *************
*/
impl<T> ContainerRepr for Vec<T> {
    type Elem = T;
}
