/*
   Appellation: actors
   Context:
   Creator: FL03 <jo3mccain@icloud.com>
   Description:
       Actors are defined to be abstract models of computation and thus represent a collection
       of complex, dynamic computational models equipped with a standard interface enabling for
       compatibility and usability
*/
pub use crate::actors::actor::*;

mod actor;

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
