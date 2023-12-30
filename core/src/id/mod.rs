/*
   Appellation: id <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # id
pub use self::{identity::*, utils::*};

pub(crate) mod identity;

pub(crate) mod utils {

    pub fn rid(length: usize) -> String {
        use rand::distributions::Alphanumeric;
        use rand::{thread_rng, Rng};

        thread_rng()
            .sample_iter(&Alphanumeric)
            .take(length)
            .map(char::from)
            .collect()
    }
}

#[cfg(test)]
mod tests {}
