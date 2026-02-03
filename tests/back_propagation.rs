use rust_micrograd::engine::Tensor;

#[test]
fn test() {
    let a = Tensor::new(2.0, "a");
    a.set_label("a");
    let b = Tensor::new(-3.0, "b");
    b.set_label("b");
    let c = Tensor::new(10.0, "c");
    c.set_label("c");
    let e = &a * b;
    e.set_label("e");
    println!("{:?}", a);
    let d = e + c;
    d.set_label("d");
    let f = Tensor::new(-2.0, "f");
    f.set_label("f");
    let L = d * f;
    L.set_label("L");
    println!("{:?}", L);
    println!("{:?}", L.prev());

    let h = 0.001;
    let L1 = L.data();
    let L2 = L.data() + h;
    let grad = (L2 - L1) / h;
    println!("L grad: {}", grad);
    // L.grad: 1.0

    // derivate L by d
    // multiplier derivative
    // dL/dd = f
    // dL/df = d
    let a = Tensor::new(2.0, "a");
    a.set_label("a");
    let b = Tensor::new(-3.0, "b");
    b.set_label("b");
    let c = Tensor::new(10.0, "c");
    c.set_label("c");
    let e = a * b;
    e.set_label("e");
    let d = e + c;
    d.set_label("d");
    d.set_data(d.data());
    let f = Tensor::new(-2.0, "f");
    f.set_label("f");
    let L1 = d * f;

    let a = Tensor::new(2.0, "a");
    a.set_label("a");
    let b = Tensor::new(-3.0, "b");
    b.set_label("b");
    let c = Tensor::new(10.0, "c");
    c.set_label("c");
    let e = a * b;
    e.set_label("e");
    let d = e + c;
    d.set_label("d");
    d.set_data(d.data() + h);
    let f = Tensor::new(-2.0, "f");
    f.set_label("f");
    let L2 = d * f;
    let grad = (L2.data() - L1.data()) / h;
    println!("dL/dd grad: {}", grad);

    // plus derivate
    // dd/dc = 1.0
    // dd/de = 1.0

    // Chain Rule
    // dL/dc = dL/dd * dd/dc

    // 단순화
    // 곱셈은 역전파 시 이전 grad에 상대의 값을 곱함
    // 덧셈은 역전파 시 이전 grad를 그대로 전달
}
