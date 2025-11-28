/*
    Appellation: depth <module>
    Created At: 2025.11.28:15:03:02
    Contrib: @FL03
*/

/// The [`NetworkDepth`] trait is used to define the depth/kind of a neural network model.
pub trait NetworkDepth {
    private!();
}

macro_rules! network_format {
    (#[$tgt:ident] $vis:vis enum {$($name:ident),* $(,)?}) => {
        $(
            network_format!(@impl #[$tgt] $vis $name);
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

network_format! {
    #[NetworkDepth]
    pub enum {
        Deep,
        Shallow,
    }
}
