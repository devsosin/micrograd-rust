use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Debug,
    hash::Hash,
    ops::{Add, Div, Mul, Neg, Sub},
    rc::Rc,
};

#[derive(PartialEq, Eq)]
enum Operation {
    None,
    Neg,
    Add,
    Mul,
    Pow,
    Tanh,
    Exp,
}

// 실제 데이터를 담고 있는 내부 구조체
struct TensorData {
    data: f64,
    grad: f64,
    label: String,

    _backward: Option<fn(&Tensor)>,
    _prev: Vec<Tensor>,
    _op: Operation,
}

// 사용자가 다룰 Tensor 구조체 (스마트 포인터 래퍼)
#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

// for hashing
// Rc가 가리키는 포인터 주소를 비교하는 방식
impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Tensor {}

impl Hash for Tensor {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

impl Tensor {
    pub fn new(data: f64, label: &str) -> Self {
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: 0.0,
            label: label.into(),
            _backward: None,
            _prev: vec![],
            _op: Operation::None,
        })))
    }

    fn new_with_operation(data: f64, operation: Operation) -> Self {
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: 0.0,
            label: String::new(),
            _backward: None,
            _prev: vec![],
            _op: operation,
        })))
    }

    // 데이터 조회
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    pub fn prev(&self) -> Vec<Tensor> {
        self.0.borrow()._prev.clone()
    }
    pub fn label(&self) -> String {
        self.0.borrow().label.clone()
    }

    // setter
    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }
    pub fn set_label(&self, label: &str) {
        self.0.borrow_mut().label = label.into()
    }

    // caller
    pub fn backward(&self) {
        let mut todos = self.topological_sort();
        todos.reverse();

        self.set_grad(1.0);

        for node in &todos {
            match node.0.borrow()._backward {
                Some(f) => f(&node),
                None => (),
            }
        }
    }

    pub fn topological_sort(&self) -> Vec<Tensor> {
        let mut visited = HashSet::new();
        let mut todo = Vec::new();

        self._build_todo(&mut visited, &mut todo);

        todo
    }

    fn _build_todo(&self, visited: &mut HashSet<Tensor>, todo: &mut Vec<Tensor>) {
        if !visited.contains(self) {
            visited.insert(self.clone());

            for p in &self.prev() {
                p._build_todo(visited, todo);
            }

            todo.push(self.clone());
        }
    }

    // activation functions -> to trait?
    pub fn tanh(&self) -> Tensor {
        let e_2x = (self.data() * 2.0).exp();
        let out = Tensor::new_with_operation((e_2x - 1.0) / (e_2x + 1.0), Operation::Tanh);
        out.0.borrow_mut()._prev = vec![self.clone()];

        fn _backward(out: &Tensor) {
            let prev = out.prev();
            for p in &prev {
                p.set_grad(p.grad() + (1.0 - out.data().powi(2)) * out.grad());
            }
        }

        out.0.borrow_mut()._backward = Some(_backward);

        out
    }

    // i64 or f64
    pub fn pow(&self, rhs: f64) -> Tensor {
        let rhs = Tensor::new(rhs, "temp");
        let out = Tensor::new_with_operation(self.data().powf(rhs.data()), Operation::Pow);
        out.0.borrow_mut()._prev = vec![self.clone(), rhs.clone()];

        fn _backward(out: &Tensor) {
            let prev = out.prev();
            let l = &prev[0];
            let r = &prev[1];

            l.set_grad(l.grad() + (r.data() * l.data().powf(r.data() - 1.0)) * out.grad());
        }

        out.0.borrow_mut()._backward = Some(_backward);

        out
    }

    pub fn exp(&self) -> Tensor {
        let x = self.data();
        let out = Tensor::new_with_operation(x.exp(), Operation::Exp);

        out.0.borrow_mut()._prev = vec![self.clone()];

        fn _backward(out: &Tensor) {
            let prev = out.prev();
            for p in &prev {
                p.set_grad(p.grad() + out.data() * out.grad());
            }
        }

        // 빠져있었음..
        out.0.borrow_mut()._backward = Some(_backward);

        out
    }

    // temporal functions
    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor {{ {} | data: {:.4} | grad: {:.4} }}",
            self.label(),
            self.data(),
            self.grad()
        )
    }
}

// ---- Add 구현
impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let out = Tensor::new_with_operation(self.data() + rhs.data(), Operation::Add);
        out.0.borrow_mut()._prev = vec![self.clone(), rhs.clone()];

        fn _backward(out: &Tensor) {
            let prev = out.prev();
            for p in &prev {
                p.set_grad(p.grad() + 1.0 * out.grad());
            }
        }

        out.0.borrow_mut()._backward = Some(_backward);

        out
    }
}

impl Add for Tensor {
    type Output = Tensor;

    // 재사용
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

// in python, radd
impl Add<f64> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::new(rhs, "temp");
        &self + &rhs
    }
}

impl Add<f64> for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::new(rhs, "temp");
        self + &rhs
    }
}

// ---- Mul 구현
impl Mul for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        let out = Tensor::new_with_operation(self.data() * rhs.data(), Operation::Mul);
        out.0.borrow_mut()._prev = vec![self.clone(), rhs.clone()];

        fn _backward(out: &Tensor) {
            let prev = out.prev();
            // TODO: TEST, 무조건 2개 나오는지
            let l = &prev[0];
            let r = &prev[1];
            l.set_grad(l.grad() + r.data() * out.grad());
            r.set_grad(r.grad() + l.data() * out.grad());
        }

        out.0.borrow_mut()._backward = Some(_backward);

        out
    }
}

impl Mul for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

impl Mul<&Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        &self * rhs
    }
}

impl Mul<Tensor> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        self * &rhs
    }
}

impl Mul<f64> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::new(rhs, "temp");
        &self * rhs
    }
}

impl Mul<f64> for &Tensor {
    type Output = Tensor;

    fn mul(self, rhs: f64) -> Self::Output {
        let rhs = Tensor::new(rhs, "temp");
        self * rhs
    }
}

// in python, rmul
impl Mul<Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        let temp = Tensor::new(self, "temp");
        temp * rhs
    }
}

impl Mul<&Tensor> for f64 {
    type Output = Tensor;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        let temp = Tensor::new(self, "temp");
        temp * rhs
    }
}

// Div
impl Div for &Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        // true div of a/b = a * b**-1
        // let rhs = Tensor::new(rhs.data().powi(-1), "temp");
        self * rhs.pow(-1.0)
    }
}

impl Div for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Self) -> Self::Output {
        &self / &rhs
    }
}

// ---- Sub
impl Neg for &Tensor {
    type Output = Tensor;

    fn neg(self) -> Self::Output {
        let out = Tensor::new_with_operation(-self.data(), Operation::Neg);
        out.0.borrow_mut()._prev = vec![self.clone()];

        // fn _backward(out: &Tensor) {
        //     let prev = out.prev();
        //     for p in prev {
        //         p.set_grad(p.grad() + -1.0 * out.grad());
        //     }
        // }
        // out.0.borrow_mut()._backward = Some(_backward);

        out
    }
}

impl Sub for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Self) -> Self::Output {
        self + &(-rhs)
    }
}

impl Sub<f64> for &Tensor {
    type Output = Tensor;

    fn sub(self, rhs: f64) -> Self::Output {
        self + -rhs
    }
}

pub trait Activation {
    fn tanh(&self) -> Tensor;
}
