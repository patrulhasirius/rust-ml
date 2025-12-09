use ndarray::{Array, ArrayView1, Axis, Dim};
use ndarray_rand::{
    RandomExt,
    rand::rngs::StdRng,
    rand_distr::{Uniform, uniform::SampleUniform},
};
use num_traits::Float;

pub type NormalArray<T> = Array<T, Dim<[usize; 2]>>;

pub trait Sigmoid {
    fn sigmoid(self) -> Self;
}

impl<T: Float> Sigmoid for T {
    fn sigmoid(self) -> Self {
        let one = T::one();

        // Formula: 1 / (1 + e^-x)
        one / (one + (-self).exp())
    }
}

#[derive(Debug, Clone)]
pub struct NN<T: Float + SampleUniform> {
    count: usize,
    shape: Vec<usize>,
    ws: Vec<NormalArray<T>>,
    bs: Vec<NormalArray<T>>,
    acs: Vec<NormalArray<T>>,
}
impl<T: Float + SampleUniform + 'static> NN<T> {
    pub fn new_random(shape: Vec<usize>, rng: &mut StdRng, low: T, high: T) -> Self {
        assert!(!shape.is_empty(), "empty shape");
        let count = shape.len() - 1;
        let mut ws: Vec<NormalArray<T>> = Vec::with_capacity(shape.len() - 1);
        let mut bs: Vec<NormalArray<T>> = Vec::with_capacity(shape.len() - 1);
        let mut acs: Vec<NormalArray<T>> = Vec::with_capacity(shape.len());

        acs.push(Array::random_using(
            (1, shape[0]),
            Uniform::<T>::new(low, high).unwrap(),
            rng,
        ));

        for i in 1..shape.len() {
            ws.push(Array::random_using(
                (acs[i - 1].ncols(), shape[i]),
                Uniform::<T>::new(low, high).unwrap(),
                rng,
            ));
            bs.push(Array::random_using(
                (1, shape[i]),
                Uniform::<T>::new(low, high).unwrap(),
                rng,
            ));
            acs.push(Array::random_using(
                (1, shape[i]),
                Uniform::<T>::new(low, high).unwrap(),
                rng,
            ));
        }
        Self {
            count,
            shape,
            ws,
            bs,
            acs,
        }
    }
    pub fn input(&mut self, input: &ArrayView1<T>) {
        self.acs[0].assign(&input.insert_axis(Axis(0)));
    }
    pub fn output(&self) -> &NormalArray<T> {
        self.acs.last().expect("Empty NN")
    }
    pub fn foward(&mut self) {
        for i in 0..self.count {
            let z = self.acs[i].dot(&self.ws[i]) + &self.bs[i];
            self.acs[i + 1] = z.mapv(Sigmoid::sigmoid);
        }
    }
    fn cost(self, ti: &NormalArray<T>, to: &NormalArray<T>) -> T {
        todo!()
    }
    fn finite_diff(&self, eps: T, ti: &NormalArray<T>, to: &NormalArray<T>) -> Self {
        todo!()
    }
}
