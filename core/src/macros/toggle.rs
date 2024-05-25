/*
    Appellation: toggle <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[macro_export]
macro_rules! toggle {
    (enum $($name:ident),* $(,)?) => {
        $(toggle!(@enum $name);)*
    };

    (@enum $name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        pub enum $name {}

        impl $crate::traits::misc::toggle::Toggle for $name {}
    };
}
