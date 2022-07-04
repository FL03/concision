/*
   Appellation: actor
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/
pub use specifications::*;

mod specifications {

    type DSConfigBuilder = config::ConfigBuilder<config::builder::DefaultState>;

    pub trait Actor<Conf = DSConfigBuilder, Data = Vec<String>> {
        fn constructor(&self, config: Conf, data: Data) -> Self
        where
            Self: Sized;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let f = |x: usize| x.pow(x.try_into().unwrap());
        assert_eq!(f(2), 4)
    }
}
