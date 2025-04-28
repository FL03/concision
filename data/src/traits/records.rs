/*
    Appellation: records <module>
    Contrib: @FL03
*/

/// This trait generically defines the basic type of dataset that can be used throughout the
/// framework.
pub trait Records {
    type Inputs;
    type Targets;

    fn inputs(&self) -> &Self::Inputs;

    fn inputs_mut(&mut self) -> &mut Self::Inputs;

    fn targets(&self) -> &Self::Targets;

    fn targets_mut(&mut self) -> &mut Self::Targets;

    fn set_inputs(&mut self, inputs: Self::Inputs) {
        *self.inputs_mut() = inputs;
    }

    fn set_targets(&mut self, targets: Self::Targets) {
        *self.targets_mut() = targets;
    }
}
