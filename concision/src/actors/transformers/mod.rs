/*
   Appellation: transformers
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

pub trait Transformable<Actor, Conf, Data> {
    fn transform(&self, actor: Actor, data: Data) -> Self
    where
        Self: Sized;
}

#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let f = |x: usize| x.pow(x.try_into().unwrap());
        assert_eq!(f(2), 4)
    }
}
