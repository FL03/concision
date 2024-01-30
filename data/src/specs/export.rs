/*
   Appellation: export <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use std::path::Path;

pub trait Export {
    fn export(&self, path: impl AsRef<Path>) -> Result<(), std::io::Error>;
}
