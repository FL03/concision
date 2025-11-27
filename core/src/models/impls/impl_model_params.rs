/*
    appellation: impl_model_params <module>
    authors: @FL03
*/
use crate::models::ModelParamsBase;

use crate::{DeepModelRepr, RawHidden};
use concision_params::ParamsBase;
use ndarray::{Data, Dimension, RawDataClone};

impl<A, S, D, H> Clone for ModelParamsBase<S, D, H, A>
where
    D: Dimension,
    H: RawHidden<S, D> + Clone,
    S: RawDataClone<Elem = A>,
    A: Clone,
{
    fn clone(&self) -> Self {
        Self {
            input: self.input().clone(),
            hidden: self.hidden().clone(),
            output: self.output().clone(),
        }
    }
}

impl<A, S, D, H> core::fmt::Debug for ModelParamsBase<S, D, H, A>
where
    D: Dimension,
    H: RawHidden<S, D> + core::fmt::Debug,
    S: Data<Elem = A>,
    A: core::fmt::Debug,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ModelParams")
            .field("input", self.input())
            .field("hidden", self.hidden())
            .field("output", self.output())
            .finish()
    }
}

impl<A, S, D, H> core::fmt::Display for ModelParamsBase<S, D, H, A>
where
    D: Dimension,
    H: RawHidden<S, D> + core::fmt::Debug,
    S: Data<Elem = A>,
    A: core::fmt::Display,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{{ input: {i}, hidden: {h:?}, output: {o} }}",
            i = self.input(),
            h = self.hidden(),
            o = self.output()
        )
    }
}

impl<A, S, D, H> core::ops::Index<usize> for ModelParamsBase<S, D, H, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    H: DeepModelRepr<S, D>,
    A: Clone,
{
    type Output = ParamsBase<S, D>;

    fn index(&self, index: usize) -> &Self::Output {
        if index == 0 {
            self.input()
        } else if index == self.count_hidden() + 1 {
            self.output()
        } else {
            &self.hidden().as_slice()[index - 1]
        }
    }
}

impl<A, S, D, H> core::ops::IndexMut<usize> for ModelParamsBase<S, D, H, A>
where
    D: Dimension,
    S: Data<Elem = A>,
    H: DeepModelRepr<S, D>,
    A: Clone,
{
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index == 0 {
            self.input_mut()
        } else if index == self.count_hidden() + 1 {
            self.output_mut()
        } else {
            &mut self.hidden_mut().as_mut_slice()[index - 1]
        }
    }
}
