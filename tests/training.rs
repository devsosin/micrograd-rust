use rust_micrograd::engine::Tensor;

#[test]
fn test() {
    let a = Tensor::new(2.0, "a"); // grad: 6.0
    a.set_label("a");
    let b = Tensor::new(-3.0, "b"); // grad: -2.0*2.0 = -4.0
    b.set_label("b");
    let c = Tensor::new(10.0, "c"); // grad: -2.0
    c.set_label("c");
    let f = Tensor::new(-2.0, "f"); // grad: 4.0
    f.set_label("f");

    // leaf nodes에서 학습률 0.01을 grad에 곱하여 data에 더해주면,
    // = L에 대한 기울기에 대해 + 방향으로 약간의 값을 더해주면,
    // L의 값이 positive 방향으로 변함 -> NN 학습의 기본 개념
    a.set_data(a.data() + 0.01 * 6.0);
    b.set_data(b.data() + 0.01 * -4.0);
    c.set_data(c.data() + 0.01 * -2.0);
    f.set_data(f.data() + 0.01 * 4.0);

    let e = &a * &b; // -6.0, grad: -2.0
    e.set_label("e");
    let d = &e + &c; // 4.0, grad: -2.0
    d.set_label("d");
    let L = &d * &f; // -8.0 (기존 값)
    L.set_label("L");

    println!("{:?}", L); // -7.286496

    // Training에서는 Loss function에 대해서 미분하므로
    // leaf node의 기울기에 대해 반대로 값을 update해주면 loss가 작아지게 할 수 있음.
}
