fn f(x: f64) -> f64 {
    3.0 * x.powi(2) - 4.0 * x + 5.0
}

#[test]
fn test() {
    // 미분 정의: 특정 한 점 x에서 h가 0에 한 없이 가까워 질 때 f(x+h)와 f(x)의 값의 차이 = 기울기
    // lim h->0
    // ( f(x+h) - f(x) ) / h
    let h = 0.0000000001;
    let x = -3.0;
    let result = (f(x + h) - f(x)) / h;

    println!("Result: {}", result);

    // inputs
    let mut a = 2.0;
    let mut b = -3.0;
    let mut c = 10.0;

    let d = a * b + c;
    println!("D: {}", d);

    let d1 = a * b + c;
    // a += h;
    // b += h;
    c += h;
    let d2 = a * b + c;

    println!("D1: {}", d1);
    println!("D2: {}", d2);
    println!("slope: {}", (d2 - d1) / h);
}
