/*
    Appellation: impl_from <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::params::ParamsBase;
use nd::prelude::*;

impl<A, S> FromIterator<(Array1<A>, Option<Array0<A>>)> for ParamsBase<S, Ix2>
where
    A: Clone + Default,
    S: DataOwned<Elem = A> + DataMut,
{
    fn from_iter<I: IntoIterator<Item = (Array1<A>, Option<Array0<A>>)>>(nodes: I) -> Self {
        let nodes = nodes.into_iter().collect::<Vec<_>>();
        let mut iter = nodes.iter();
        let node = iter.next().unwrap();
        let shape = Features::new(node.0.len(), nodes.len());
        let mut params = ParamsBase::default::<Biased>(shape);
        params.set_node(0, node.clone());
        for (i, node) in iter.into_iter().enumerate() {
            params.set_node(i + 1, node.clone());
        }
        params
    }
}

macro_rules! impl_from {


    (A) => {
        impl<A> From<(Array1<A>, A)> for ParamsBase<OwnedRepr<A>, Ix1>
        where
            A: Clone,
        {
            fn from((weight, bias): (Array1<A>, A)) -> Self {
                let bias = ArrayBase::from_elem((), bias);
                Self {
                    bias: Some(bias),
                    weights: weight,
                }
            }
        }
        impl<A> From<(Array1<A>, Option<A>)> for ParamsBase<OwnedRepr<A>, Ix1>
        where
            A: Clone,
        {
            fn from((weights, bias): (Array1<A>, Option<A>)) -> Self {
                Self {
                    bias: bias.map(|b| ArrayBase::from_elem((), b)),
                    weights,
                }
            }
        }
    };
    ($($bias:ty),*) => {
        $(impl_from!(@impl $bias);)*

    };
    (@impl $b:ty) => {
        impl<A, S, D> From<(ArrayBase<S, D>, Option<$b>)> for ParamsBase<S, D>
        where
            D: RemoveAxis,
            S: RawData<Elem = A>,
        {
            fn from((weights, bias): (ArrayBase<S, D>, Option<$b>)) -> Self {
                Self {
                    bias,
                    weights,
                }
            }
        }

        impl<A, S, D> From<(ArrayBase<S, D>, $b)> for ParamsBase<S, D>
        where
            D: RemoveAxis,
            S: RawData<Elem = A>,
        {
            fn from((weights, bias): (ArrayBase<S, D>, $b)) -> Self {
                Self {
                    bias: Some(bias),
                    weights,
                }
            }
        }
    };
}

impl_from!(A);
impl_from!(ArrayBase<S, D::Smaller>);