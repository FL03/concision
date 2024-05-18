/*
    Appellation: toggle <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Toggle: 'static {}

pub trait OfType {
    fn of<T>() -> bool
    where
        T: 'static,
        Self: 'static,
    {
        core::any::TypeId::of::<T>() == core::any::TypeId::of::<Self>()
    }
}

/*
 ************* Implementations *************
*/
impl<T> OfType for T {}

macro_rules! impl_toggle {
    ($($scope:ident$(<$T:ident>)?),* $(,)?) => {
        $(impl_toggle!(@impl $scope$(<$T>)?);)*
    };
    (@impl $scope:ident$(<$T:ident>)?) => {
        impl$(<$T>)? Toggle for $scope$(<$T> where $T: 'static)? {}
    };
}

impl_toggle!(
    bool,
    char,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    Option<T>
);
