use rust_micrograd::engine::Tensor;

#[test]
pub fn test() {
    // inputs
    let x1 = Tensor::new(2.0, "x1");
    let x2 = Tensor::new(0.0, "x2");

    // weights
    let w1 = Tensor::new(-3.0, "w1");
    let w2 = Tensor::new(1.0, "w2");
    // bias
    let b = Tensor::new(6.88137358, "b");

    // x1*w1 + x2*w2 + b
    let x1w1 = &x1 * &w1;
    let x2w2 = &x2 * &w2;
    let x1w1x2w2 = &x1w1 + &x2w2;
    let n = &x1w1x2w2 + &b;
    println!("n: {:?}", n);

    let o = n.tanh();
    println!("o: {:?}", o);

    println!("\n{}\n", "-".repeat(36));

    // back propagation
    // o.grad = 1.0
    o.set_grad(1.0);
    println!("o: {:?}", o);

    // o = tanh(n)
    // dtanh / dx = 1- tanh^2(x)
    //  = 1 - o**2
    n.set_grad(1.0 - o.data().powi(2)); // 0.5
    println!("n: {:?}", n);

    // plus operation passes the previous grad.
    x1w1x2w2.set_grad(0.5);
    b.set_grad(0.5);
    x1w1.set_grad(0.5);
    x2w2.set_grad(0.5);

    // multiply operation multiply previous grad with other side value.
    x1.set_grad(w1.data() * x1w1.grad());
    w1.set_grad(x1.data() * x1w1.grad());
    x2.set_grad(w2.data() * x2w2.grad());
    w2.set_grad(x2.data() * x2w2.grad());

    println!("w1: {:?}", w1);
    println!("x1: {:?}", x1);
    println!("w2: {:?}", w2);
    println!("x2: {:?}", x2);
    println!("b: {:?}", b);

    // DO normalization
}

#[test]
pub fn test_backward() {
    // inputs
    let x1 = Tensor::new(2.0, "x1");
    let x2 = Tensor::new(0.0, "x2");

    // weights
    let w1 = Tensor::new(-3.0, "w1");
    let w2 = Tensor::new(1.0, "w2");
    // bias
    let b = Tensor::new(6.88137358, "b");

    // x1*w1 + x2*w2 + b
    let x1w1 = &x1 * &w1;
    let x2w2 = &x2 * &w2;
    let x1w1x2w2 = &x1w1 + &x2w2;
    let n = &x1w1x2w2 + &b;
    println!("n: {:?}", n);

    let o = n.tanh();
    println!("o: {:?}", o);

    println!("\n{}\n", "-".repeat(36));

    // base case -> if none -> 1.0?
    o.set_grad(1.0);
    println!("o: {:?}", o);

    o.backward();
    println!("n: {:?}", n);

    n.backward();
    println!("x1w1x2w2: {:?}", x1w1x2w2);
    println!("b: {:?}", b);
    b.backward(); // do nothing (leaf node)

    x1w1x2w2.backward();
    println!("x1w1: {:?}", x1w1);
    println!("x2w2: {:?}", x2w2);
    x1w1.backward();
    println!("x1: {:?}", x1);
    println!("w1: {:?}", w1);
    x2w2.backward();
    println!("x2: {:?}", x2);
    println!("w2: {:?}", w2);

    // base grad 1.0 (last node)
    // topological sort -> in reverse order
}

#[test]
pub fn test_back_propagation() {
    // inputs
    let x1 = Tensor::new(2.0, "x1");
    let x2 = Tensor::new(0.0, "x2");

    // weights
    let w1 = Tensor::new(-3.0, "w1");
    let w2 = Tensor::new(1.0, "w2");
    // bias
    let b = Tensor::new(6.88137358, "b");

    // x1*w1 + x2*w2 + b
    let x1w1 = &x1 * &w1;
    x1w1.set_label("x1w1");
    let x2w2 = &x2 * &w2;
    x2w2.set_label("x2w2");
    let x1w1x2w2 = &x1w1 + &x2w2;
    x1w1x2w2.set_label("x1w1x2w2");
    let n = &x1w1x2w2 + &b;
    n.set_label("n");
    println!("n: {:?}", n);

    let o = n.tanh();
    o.set_label("o");
    println!("o: {:?}", o);

    println!("\n{}\n", "-".repeat(36));

    o.backward();
    println!("o: {:?}", o);
    println!("n: {:?}", n);
    println!("x1w1x2w2: {:?}", x1w1x2w2);
    println!("b: {:?}", b);
    println!("x1w1: {:?}", x1w1);
    println!("x2w2: {:?}", x2w2);
    println!("x1: {:?}", x1);
    println!("w1: {:?}", w1);
    println!("x2: {:?}", x2);
    println!("w2: {:?}", w2);
}
