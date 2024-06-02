/*
    Appellation: toggle <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[macro_export]
macro_rules! toggle {
    ($vis:vis enum {$($name:ident),* $(,)?}) => {
        $(toggle!(@impl $vis enum $name);)*
    };

    (@impl $vis:vis enum $name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        $vis enum $name {}

        impl $name {
            $vis fn of<T: 'static>() -> bool {
                use ::core::any::TypeId;
                TypeId::of::<T>() == TypeId::of::<Self>()
            }
            $vis fn phantom() -> core::marker::PhantomData<Self> {
                core::marker::PhantomData
            }
        }

        impl $crate::traits::TypeTag for $name {}
    };
}
