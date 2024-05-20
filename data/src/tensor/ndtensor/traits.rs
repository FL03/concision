/*
   Appellation: traits <module>
   Contrib: FL03 <jo3mccain@icloud.com>
*/

pub trait TensorData {
    type Elem;

    fn as_slice(&self) -> &[Self::Elem];

    fn as_mut_slice(&mut self) -> &mut [Self::Elem];
}

pub trait NdContainer<T> {
    const RANK: Option<usize> = None;

    type Data: TensorData<Elem = T>;
    type Dim;

    fn data(&self) -> &Self::Data;

    fn data_mut(&mut self) -> &mut Self::Data;

    fn dim(&self) -> Self::Dim;

    fn rank(&self) -> usize;

    fn shape(&self) -> &[usize];
}

/*
 ******** implementations ********
*/
