/*
    Appellation: toggle <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait Toggle: 'static {}

pub trait Mode: Toggle {

    fn of<K>() -> bool
    where
        K: Toggle,
    {
        core::any::TypeId::of::<Self>() == core::any::TypeId::of::<K>()
    }
}

/*
    ************* Implementations *************
*/
macro_rules! impl_toggle {
    ($($scope:ident$(<$T:ident>)?),* $(,)?) => {
        $(impl_toggle!(@impl $scope$(<$T>)?);)*
    };
    (@impl $scope:ident$(<$T:ident>)?) => {
        impl$(<$T>)? Toggle for $scope$(<$T> where $T: 'static)? {}
    };
}

impl_toggle!(bool, char, i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, Option<T>);