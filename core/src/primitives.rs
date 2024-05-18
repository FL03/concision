/*
    Appellation: primitives <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use consts::*;




pub mod consts {
    /// The default model size for any given model
    pub const D_MODEL: usize = 512;
    /// The default epsilon value for floating point operations
    pub const EPSILON: f64 = 1e-5;
}

#[allow(unused)]
pub(crate) mod rust {
    pub(crate) use core::*;
    #[cfg(all(feature = "alloc", no_std))]
    pub(crate) use self::no_std::*;
    #[cfg(feature = "std")]
    pub(crate) use self::with_std::*;
    #[cfg(all(feature = "alloc", no_std))]
    mod no_std {
        pub use alloc::borrow::Cow;
        pub use alloc::boxed::{self, Box};
        pub use alloc::collections::{self, BTreeMap, BTreeSet, BinaryHeap, VecDeque};
        pub use alloc::vec::{self, Vec};
    }
    #[cfg(feature = "std")]
    mod with_std {
        pub use std::borrow::Cow;
        pub use std::boxed::{self, Box};
        pub use std::collections::{self, BTreeMap, BTreeSet, BinaryHeap, VecDeque};
        pub use std::sync::Arc;
        pub use std::vec::{self, Vec};

        
    }

    #[cfg(all(feature = "alloc", no_std))]
    pub type Map<K, V> = alloc::collections::BTreeMap<K, V>;
    #[cfg(feature = "std")]
    pub type Map<K, V> = std::collections::HashMap<K, V>;
}
