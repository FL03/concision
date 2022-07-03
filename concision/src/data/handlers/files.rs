/*
   Appellation: csv
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/
pub use specifications::*;
pub use utils::*;

pub enum FileExtensions<Address = String> {
    Csv(Address),
    Json(Address)
}

#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FileHandler {
    pub address: String,
}

impl std::fmt::Display for FileHandler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FileHandler(address={})", self.address)
    }
}

mod specifications {
    pub trait FileSpec: Sized {
        type Address;
        type Container;
        type Data;

        fn aggregate(&self, sources: Self::Address) -> Self::Container
        where
            Self::Container: Sized;
    }
}

mod utils {
    use super::*;
    use std::io::prelude::*;

    pub fn create_file(source: String) -> std::io::Result<std::fs::File> {
        let mut file = std::fs::File::create(source)?;
        Ok(file)
    }

    pub fn read_file(source: String) -> std::io::Result<String> {
        let mut file = std::fs::File::open(source)?;
        let mut contents = String::new();
        file.read_to_string(&mut contents)?;
        Ok(contents)
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test() {
        let f = |x: usize, y: usize| x.pow(y.try_into().unwrap());
        assert_eq!(f(10, 2), 100)
    }
}
