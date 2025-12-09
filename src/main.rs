use ndarray::array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use rust_ml::NN;

#[rustfmt::skip]
const TRAIN: [(f32, f32, f32);4] = [
    (0., 0., 0.),
    (1., 0., 1.),
    (0., 1., 1.),
    (1., 1., 0.),
    ];
const SEED: u64 = 69;
const EPS: f32 = 1e-3;
const RATE: f32 = 1e-1;

fn main() {
    //let mut rng: StdRng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut rng: StdRng = StdRng::from_os_rng();
    let train_set = array![[0., 0., 0.], [1., 0., 1.], [0., 1., 1.], [1., 1., 0.],];
    let mut nn = NN::<f32>::new_random(vec![2, 2, 1], &mut rng, 0., 1.);
    let binding = train_set.row(1);
    let ti = binding.slice(ndarray::s![..-1]);
    nn.input(&ti);
    nn.foward();
    let ti_a = train_set.slice(ndarray::s![.., 0..-1]).into_owned();
    let to_a = train_set.slice(ndarray::s![.., -1]).into_owned();
    dbg!(to_a);
    println!("{}", nn.output());
    //dbg!(nn);
}
