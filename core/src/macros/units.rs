/*
    Appellation: units <module>
    Created At: 2025.12.08:19:44:33
    Contrib: @FL03
*/

macro_rules! type_tags {
    (#[$tgt:ident] $vis:vis $s:ident {$($name:ident),* $(,)?}) => {
        $(
            type_tags!(@impl #[$tgt] $vis $s $name);
        )*
    };
    (@impl #[$tgt:ident] $vis:vis enum $name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        $vis enum $name {}

        impl $tgt for $name {
            seal!();
        }
    };
    (@impl #[$tgt:ident] $vis:vis struct $name:ident) => {
        #[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        $vis struct $name;

        impl $tgt for $name {
            seal!();
        }
    };
}
