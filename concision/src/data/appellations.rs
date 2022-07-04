/*
    Appellation: appellations
    Context: module
    Creator: FL03 <jo3mccain@icloud.com>
    Description:
        An appellation is defined as a name or title or as the act of giving something a name or
        title. This leads us to create a solid foundation by discovering unique and repeatable
        methods for determining the most adequate name convention to be used by the system.
 */

#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Appellations {
    Aliens {
        prefix: String,
        first: String,
        middle: String,
        last: String,
        suffix: String
    },
    Applications {
        name: String,
        slug: String
    },

}

pub trait Appellation<Dt = String> {
    fn create(&self, name: Dt) -> Self where Self: Sized;
}


#[cfg(test)]
mod tests {
    #[test]
    fn test() {
        let f = |x: usize| x.pow(x.try_into().unwrap());
        assert_eq!(f(2), 4)
    }
}