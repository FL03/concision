/*
    Appellation: contexts
    Context:
    Creator: FL03 <jo3mccain@icloud.com>
    Description:
        ... Summary ...
 */


pub trait Contextual<
    Actor = String,
    Config = config::ConfigBuilder<config::builder::DefaultState>,
    Data = String
    > {

    fn authenticate(&self, actor: Actor) -> Actor where Self: Sized;
    fn constructor(&self, actor: Actor, config: Config, data: Data) -> Config where Self: Sized;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let c: Box<dyn super::Contextual>;
        let f = |x: usize| x.pow(x.try_into().unwrap());
        assert_eq!(f(2), 4)
    }
}