use ndarray::array;
use ndarray_rand::rand::SeedableRng;
use ndarray_rand::rand::rngs::StdRng;
use rust_ml::NN;

#[rustfmt::skip]
// XOR
const XOR: [[f32; 3];4] = [
    [0., 0., 0.],
    [1., 0., 1.],
    [0., 1., 1.],
    [1., 1., 0.],
];
#[rustfmt::skip]
// XOR
const OR: [[f32; 3];4] = [
    [0., 0., 0.],
    [1., 0., 1.],
    [0., 1., 1.],
    [1., 1., 1.],
];
const SEED: u64 = 69;
const EPS: f32 = 1e-1;
const RATE: f32 = 1e-1;

fn main() {
    //let mut rng: StdRng = rand::rngs::StdRng::seed_from_u64(SEED);
    let mut rng: StdRng = StdRng::from_os_rng();
    let rate = RATE;
    let eps = EPS;
    let train_set = ndarray::arr2(&XOR);
    let mut nn = NN::<f32>::new_random(vec![2, 2, 1], &mut rng, -1., 1.);
    let ti_a = train_set.slice(ndarray::s![.., ..-1]).into_owned();
    let to_a = train_set.slice(ndarray::s![.., -1..=-1]).into_owned();
    dbg!(&ti_a);
    dbg!(&to_a);
    println!("{}", nn.cost(&ti_a, &to_a));
    nn.learn(rate, eps, &ti_a, &to_a);
    println!("{}", nn.cost(&ti_a, &to_a));
    //dbg!(nn);

    (0..1000 * 100).for_each(|_| nn.learn(rate, eps, &ti_a, &to_a));
    println!("{}", nn.cost(&ti_a, &to_a));
    for i in 0..ti_a.nrows() {
        let ti = ti_a.row(i);
        nn.input(&ti);
        nn.foward();
        println!("{} = {}", ti, nn.output());
    }
}
