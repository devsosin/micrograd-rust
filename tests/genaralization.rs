use rust_micrograd::engine::Tensor;

#[test]
fn test() {
    let a = Tensor::new_with_label(1.0, "a");
    let b = &a + 1.0;
    println!("b: {:?}", b);
    let b = &a * 2.0;
    println!("b: {:?}", b);

    let b = 2.0 * a;
    println!("b: {:?}", b);
}
