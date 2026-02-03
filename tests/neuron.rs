use std::iter::zip;

use rust_micrograd::{
    engine::Tensor,
    nn::{Layer, MLP, Module, Neuron},
};

#[test]
fn test_neuron() {
    let neuron = Neuron::new(2);

    println!("weights: {:?}", neuron.weights());
    println!("bias: {:?}", neuron.bias());

    let x = vec![2.0, 3.0];
    let x = Tensor::from_vec(x);

    let o = neuron.forward(&x);
    println!("output: {:?}", o);
}

#[test]
fn test_layer() {
    let x = vec![2.0, 3.0];
    let x = Tensor::from_vec(x);

    let layer = Layer::new(2, 3);
    let out = layer.forward(&x);

    println!("out: {:?}", out);
}

#[test]
fn test_mlp() {
    let x = vec![2.0, 3.0, -1.0];
    let x = Tensor::from_vec(x);

    let mlp = MLP::new(3, vec![4, 4, 1]);
    let out = mlp.forward(&x);

    println!("out: {:?}", out);
}

#[test]
fn test_back_propagation() {
    let xs = vec![
        Tensor::from_vec(vec![2.0, 3.0, -1.0]),
        Tensor::from_vec(vec![3.0, -1.0, 0.5]),
        Tensor::from_vec(vec![0.5, 1.0, 1.0]),
        Tensor::from_vec(vec![1.0, 1.0, -1.0]),
    ];

    let ys = vec![1.0, -1.0, -1.0, 1.0];
    let n = MLP::new(3, vec![4, 4, 1]);
    let y_preds = xs
        .iter()
        .map(|x| n.forward(x))
        .collect::<Vec<Vec<Tensor>>>();
    println!("---- Y_Predictions ----\n{:?}\n\n", y_preds);

    let costs: Vec<Tensor> = zip(&ys, y_preds)
        .map(|(y, y_pred)| (&y_pred[0] - *y).pow(2.0))
        .collect();
    println!("---- Costs ----\n{:?}\n\n", costs);
    let mut loss = costs.iter().fold(Tensor::new(0.0), |acc, x| &acc + x);
    println!("---- Loss ----\n{:?}\n\n", loss);

    loss.backward();
    println!("weights[0]: {:?}", n.layers()[0].neurons()[0].weights()[0]);

    let mut params = n.parameters();
    params.reverse();

    println!(
        "---- Params ----\n{:?}\n\n",
        params.iter().map(|p| p.len() as usize).sum::<usize>()
    );

    println!("\n{}\n", "-".repeat(36));

    let leraning_rate = 0.01;

    for layer_params in &params {
        for p in layer_params {
            p.set_data(p.data() - leraning_rate * p.grad());
        }
    }

    // Training
    for idx in 0..20 {
        // feed forward
        let y_preds = xs
            .iter()
            .map(|x| n.forward(x))
            .collect::<Vec<Vec<Tensor>>>();

        loss = zip(&ys, y_preds)
            .map(|(y, y_pred)| (&y_pred[0] - *y).pow(2.0))
            .sum();

        // zero_grad
        // for layer_params in &n.parameters() {
        //     for p in layer_params {
        //         p.set_grad(0.0);
        //     }
        // }
        n.zero_grad();

        // backward
        loss.backward();

        // update
        for layer_params in &n.parameters() {
            for p in layer_params {
                p.set_data(p.data() - leraning_rate * p.grad());
            }
        }
        println!("[{}] loss: {}", idx, loss.data());
    }

    println!("---- Loss ----\n{:?}\n\n", loss);
    let y_preds = xs
        .iter()
        .map(|x| n.forward(x))
        .collect::<Vec<Vec<Tensor>>>();
    println!("---- Y_Predictions ----\n{:?}\n\n", y_preds);
    // real y: 1.0, -1.0, -1.0, 1.0
}
