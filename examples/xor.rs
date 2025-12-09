use derive_more::{Mul, Sub};
use ndarray::{Array, Dim, array};
use ndarray::{ArrayBase, OwnedRepr};
use ndarray_rand::RandomExt;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use ndarray_rand::rand_distr::Uniform;
use rust_ml::NormalArray;
use rust_ml::Sigmoid;

const SEED: u64 = 69;
const EPS: f32 = 1e-1;
const RATE: f32 = 1e-1;

#[derive(Debug, Clone, Sub, Mul)]
struct Xor {
    a0: NormalArray<f32>,
    w1: NormalArray<f32>,
    b1: NormalArray<f32>,
    w2: NormalArray<f32>,
    b2: NormalArray<f32>,
}
impl Xor {
    fn new(
        a0: NormalArray<f32>,
        w1: NormalArray<f32>,
        b1: NormalArray<f32>,
        w2: NormalArray<f32>,
        b2: NormalArray<f32>,
    ) -> Self {
        Xor { a0, w1, b1, w2, b2 }
    }
    fn foward(&self) -> ArrayBase<OwnedRepr<f32>, Dim<[usize; 1]>, f32> {
        let s = self.clone();
        let mut a1 = s.a0.dot(&s.w1);
        a1 = a1 + s.b1;
        a1 = a1.mapv_into(Sigmoid::sigmoid);

        let mut a2 = a1.dot(&s.w2);
        a2 = a2 + s.b2;
        a2 = a2.mapv_into(Sigmoid::sigmoid);
        a2.row(0).to_owned()
    }
    fn cost(&self, train_set: &NormalArray<f32>) -> f32 {
        let mut m = self.clone();
        train_set.rows().into_iter().fold(0., |acc, row| {
            let ti = row.slice(ndarray::s![..-1]);
            let to = row.last().unwrap();
            m.a0.row_mut(0).assign(&ti.to_owned());
            let result = m.foward();
            acc + (result.first().unwrap() - to).powi(2)
        }) / train_set.nrows() as f32
    }
    fn finite_diff(&self, train_set: &NormalArray<f32>) -> Self {
        // 1. Create a copy to hold the FINAL gradients
        let mut grads = self.clone();
        // 2. Create a working copy to PERTURB weights
        let mut scanner = self.clone();
        // 3. Calculate base cost once
        let c = self.cost(train_set);
        for (idx, _) in self.w1.indexed_iter() {
            let saved = scanner.w1[idx];
            scanner.w1[idx] += EPS;
            let new_cost = scanner.cost(train_set);
            scanner.w1[idx] = saved;
            grads.w1[idx] = (new_cost - c) / EPS;
        }
        for (idx, _) in self.b1.indexed_iter() {
            let saved = scanner.b1[idx];
            scanner.b1[idx] += EPS;
            let new_cost = scanner.cost(train_set);
            scanner.b1[idx] = saved;
            grads.b1[idx] = (new_cost - c) / EPS;
        }
        for (idx, _) in self.w2.indexed_iter() {
            let saved = scanner.w2[idx];
            scanner.w2[idx] += EPS;
            let new_cost = scanner.cost(train_set);
            scanner.w2[idx] = saved;
            grads.w2[idx] = (new_cost - c) / EPS;
        }
        for (idx, _) in self.b2.indexed_iter() {
            let saved = scanner.b2[idx];
            scanner.b2[idx] += EPS;
            let new_cost = scanner.cost(train_set);
            scanner.b2[idx] = saved;
            grads.b2[idx] = (new_cost - c) / EPS;
        }

        grads
    }
    fn learn(&mut self, train_set: &NormalArray<f32>) {
        let g = self.finite_diff(train_set);
        *self = self.clone() - g * RATE;
    }
}

fn main() {
    //let mut rng: StdRng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut rng: StdRng = StdRng::from_os_rng();
    let train_set = array![[0., 0., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 0.],];
    // let training_output = Array::from_shape_vec((1, 4), vec![0., 1., 1., 0.]).unwrap();
    let a0 = array![[0., 1.]];
    let w1 = Array::random_using((2, 2), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);
    let b1 = Array::random_using((1, 2), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);

    let w2 = Array::random_using((2, 1), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);
    let b2 = Array::random_using((1, 1), Uniform::<f32>::new(0., 1.).unwrap(), &mut rng);

    let mut m = Xor::new(a0, w1, b1, w2, b2);

    dbg!(&m);

    println!("cost = {}", m.cost(&train_set));
    for _ in 0..(1000 * 100) {
        m.learn(&train_set);
        //println!("cost = {}", m.cost(&train_set));
    }
    train_set.rows().into_iter().for_each(|row| {
        let ti = row.slice(ndarray::s![..-1]);
        let to = row.last().unwrap();
        m.a0.row_mut(0).assign(&ti.to_owned());
        let result = m.foward();
        println!("{} {}", ti, result);
    })
}
