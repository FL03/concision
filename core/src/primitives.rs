/*
    Appellation: primitives <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::constants::*;

mod constants {
    pub const DEFAULT_MODEL_SIZE: usize = 2048;
}

#[allow(unused_imports)]
pub(crate) mod rust {
    pub(crate) use core::*;

    #[cfg(no_std)]
    pub(crate) use self::no_std::*;
    #[cfg(feature = "std")]
    pub(crate) use self::with_std::*;

    #[cfg(no_std)]
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
        pub(crate) use std::sync::Arc;
        pub use std::vec::{self, Vec};
    }

    #[cfg(no_std)]
    pub type Map<K, V> = collections::BTreeMap<K, V>;
    #[cfg(feature = "std")]
    pub type Map<K, V> = collections::HashMap<K, V>;
}
