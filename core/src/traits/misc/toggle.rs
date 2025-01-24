/*
    Appellation: toggle <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait TypeTag: 'static {}

pub trait TypeOf: 'static {
    fn of<T>() -> bool
    where
        T: 'static,
        Self: 'static,
    {
        core::any::TypeId::of::<T>() == core::any::TypeId::of::<Self>()
    }

    fn is<T>() -> bool
    where
        T: 'static,
        Self: 'static,
    {
        core::any::TypeId::of::<Self>() == core::any::TypeId::of::<T>()
    }
}

/*
 ************* Implementations *************
*/
impl<T: 'static> TypeOf for T {}
