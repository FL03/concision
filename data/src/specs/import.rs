/*
   Appellation: import <mod>
   Contrib: FL03 <jo3mccain@icloud.com>
*/
use std::path::Path;

pub trait Import {
    type Obj;

    fn import(&mut self, path: impl AsRef<Path>) -> Result<Self::Obj, std::io::Error>;
}
