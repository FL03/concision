/*
   Appellation: files
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/
pub use specifications::*;
pub use utils::*;

/// A collection of file extensions with variable a data type; defaults to String
pub enum FileExtensions<Data = String> {
    Csv(Data),
    Json(Data),
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

    pub trait FileSpec<Addr = String, Cache = bool, Cont = Vec<String>, Data = String> {
        fn aggregate(&self, address: Addr) -> Cont
        where
            Self: Sized;
        fn create(&self, address: Addr, data: Data) -> std::io::Result<std::fs::File>
        where
            Self: Sized;
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
        let a: Box<dyn super::FileSpec>;
        let f = |x: usize, y: usize| x.pow(y.try_into().unwrap());
        assert_eq!(f(10, 2), 100)
    }
}
