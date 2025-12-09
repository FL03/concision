/*
    Appellation: units <module>
    Created At: 2025.12.08:19:44:33
    Contrib: @FL03
*/

macro_rules! unit_types {
    (#[$tgt:ident] $vis:vis enum {$($name:ident),* $(,)?}) => {
        $(
            unit_types!(@impl #[$tgt] $vis $name);
        )*
    };
    (@impl #[$tgt:ident] $vis:vis $name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        $vis enum $name {}

        impl $tgt for $name {
            seal!();
        }
    };
}
