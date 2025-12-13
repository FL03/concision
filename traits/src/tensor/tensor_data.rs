/*
    Appellation: repr <module>
    Created At: 2025.12.09:10:04:11
    Contrib: @FL03
*/

pub trait RawTensorData {
    type Elem;

    private! {}
}

/*
 ************* Implementations *************
*/

impl<A, S> RawTensorData for S
where
    S: ndarray::RawData<Elem = A>,
{
    type Elem = A;

    seal! {}
}
