/*
    Appellation: norm <module>
    Contrib: @FL03
*/
/// a trait for computing the L1 norm of a tensor or array
pub trait L1Norm {
    type Output;

    fn l1_norm(&self) -> Self::Output;
}
/// a trait for computing the L2 norm of a tensor or array
pub trait L2Norm {
    type Output;
    /// compute the L2 norm of the tensor or array
    fn l2_norm(&self) -> Self::Output;
}

/// The [Norm] trait serves as a unified interface for various normalization routnines. At the
/// moment, the trait provides L1 and L2 techniques.
pub trait Norm {
    type Output;
    /// compute the L1 norm of the tensor or array
    fn l1_norm(&self) -> Self::Output;
    /// compute the L2 norm of the tensor or array
    fn l2_norm(&self) -> Self::Output;
}

/*
 ************* Implementations *************
*/
impl<U, V> Norm for U
where
    U: L1Norm<Output = V> + L2Norm<Output = V>,
{
    type Output = V;

    fn l1_norm(&self) -> Self::Output {
        <Self as L1Norm>::l1_norm(self)
    }

    fn l2_norm(&self) -> Self::Output {
        <Self as L2Norm>::l2_norm(self)
    }
}

macro_rules! impl_norm {
    ($trait:ident::$method:ident($($param:ident: $type:ty),*) => $self:ident$(.$call:ident())*) => {
        impl<A, S, D> $trait for ndarray::ArrayBase<S, D, A>
        where
            A: 'static + Clone + num_traits::Float,
            D: ndarray::Dimension,
            S: ndarray::Data<Elem = A>,
        {
            type Output = A;

            fn $method(&self, $($param: $type),*) -> Self::Output {
                self$(.$call())*
            }
        }

        impl<'a, A, S, D> $trait for &'a ndarray::ArrayBase<S, D, A>
        where
            A: 'static + Clone + num_traits::Float,
            D: ndarray::Dimension,
            S: ndarray::Data<Elem = A>,
        {
            type Output = A;

            fn $method(&self, $($param: $type),*) -> Self::Output {
                self$(.$call())*
            }
        }

        impl<'a, A, S, D> $trait for &'a mut ndarray::ArrayBase<S, D, A>
        where
            A: 'static + Clone + num_traits::Float,
            D: ndarray::Dimension,
            S: ndarray::Data<Elem = A>,
        {
            type Output = A;

            fn $method(&self, $($param: $type),*) -> Self::Output {
                self$(.$call())*
            }
        }
    };
}

impl_norm! { L2Norm::l2_norm() => self.pow2().sum().sqrt() }

impl_norm! { L1Norm::l1_norm() => self.abs().sum() }
