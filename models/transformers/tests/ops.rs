/*
    Appellation: ops <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate concision_core as concision;
extern crate concision_transformers as transformers;

use concision::prelude::{linarr, NdResult};
use ndarray::prelude::*;
use ndarray::Order;
use transformers::ops::*;

fn order(row_major: bool) -> Order {
    if row_major {
        Order::RowMajor
    } else {
        Order::ColumnMajor
    }
}

fn merge3<T>(heads: &Array3<T>, row_major: bool) -> NdResult<Array2<T>>
where
    T: Clone,
{
    let (n, seq, query) = heads.dim();
    let mut tmp = heads.clone();
    // swap the head and sequence axes
    tmp.swap_axes(0, 1);
    // reshape the qkv matrix into a 2d array
    tmp.to_shape(((seq, n * query), order(row_major)))
        .map(|x| x.to_owned())
}

fn merge4<T>(heads: &Array4<T>, row_major: bool) -> NdResult<Array3<T>>
where
    T: Clone,
{
    let (batch, n, seq, query) = heads.dim();
    let mut tmp = heads.clone();
    // swap the head and sequence axes
    tmp.swap_axes(1, 2);
    // reshape the qkv matrix into a 2d array
    tmp.to_shape(((batch, seq, n * query), order(row_major)))
        .map(|x| x.to_owned())
}

#[test]
fn test_merge() {
    let shape = (3, 4, 5);
    let dout = (4, 15);
    let arr = linarr::<f64, Ix3>(shape.clone()).unwrap();
    let a = arr.clone().merge().unwrap();

    assert_eq!(a.dim(), dout);
    assert_eq!(a, merge3(&arr, false).unwrap());
}

#[test]
fn test_merge_batch() {
    let shape = (2, 3, 4, 5);
    let dout = (2, 4, 15);
    let arr = linarr::<f64, Ix4>(shape).unwrap();
    let a = arr.merge().unwrap();

    assert_eq!(a.dim(), dout);
    assert_eq!(a, merge4(&arr, false).unwrap());
}

#[test]
fn test_split() {
    let heads = 2;
    let shape = (3, 4, 6);
    let dout = (3, heads, 4, 3);
    let arr = linarr::<f64, Ix3>(shape).unwrap();
    let a = arr.split(heads).unwrap();

    assert_eq!(a.dim(), dout);
}

#[test]
#[ignore = "Needs to be fixed; currently fails when trying to recreate the original data."]
fn reshape_ops() {
    let heads = 2;
    let dim_input = (2, 4, 6); // (batch, seq, model)
    let dim_split = (2, heads, 4, 3); // (batch, heads, seq, model)
    let data = linarr::<f64, Ix3>(dim_input).unwrap();

    let a = data.split(heads).unwrap(); // split_batch(&data, heads).unwrap();
    let b = a.merge().unwrap(); // merge_batch(&a).unwrap();

    assert_eq!(a.dim(), dim_split);
    assert_eq!(b.dim(), dim_input);
    assert_eq!(b, data);
    // for (i, &j) in data.split(heads).unwrap().indexed_iter() {
    //     assert_eq!(j, a[i]);
    // }
    // for (i, &j) in b.indexed_iter() {
    //     assert_eq!(j, data[i]);
    // }
}
