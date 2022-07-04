/*
   Appellation: automata
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       ... Summary ...
*/

#[derive(Clone, Debug, Hash, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum AutomataStates {
    Aggregating,
    Computing,
    Determining,
    Terminating,
}

pub trait Automata<Alpha, Beta, Lambda, Gamma, Dirac> {
    fn constructor(
        &self,
        alpha: Alpha,
        beta: Beta,
        lambda: Lambda,
        gamma: Gamma,
        dirac: Dirac,
    ) -> Self
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
