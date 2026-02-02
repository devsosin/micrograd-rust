use std::{
    cell::RefCell,
    fmt::Debug,
    ops::{Add, Mul},
    rc::Rc,
};

// 실제 데이터를 담고 있는 내부 구조체
struct TensorData {
    data: f64,
    grad: f64,
    label: String,
    _prev: Vec<Tensor>,
}

// 사용자가 다룰 Tensor 구조체 (스마트 포인터 래퍼)
#[derive(Clone)]
pub struct Tensor(Rc<RefCell<TensorData>>);

impl Tensor {
    pub fn new(data: f64) -> Self {
        Tensor(Rc::new(RefCell::new(TensorData {
            data,
            grad: 0.0,
            _prev: vec![],
            label: String::new(),
        })))
    }

    // 데이터 조회
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    pub fn label(&self) -> String {
        self.0.borrow().label.clone()
    }

    pub fn set_label(&self, label: &str) {
        self.0.borrow_mut().label = label.into()
    }
    pub fn set_grad(&self, grad: f64) {
        self.0.borrow_mut().grad = grad;
    }

    // temporal functions
    pub fn prev(&self) -> Vec<Tensor> {
        self.0.borrow()._prev.clone()
    }
    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor {{ {} | data: {} | grad: {} }}",
            self.label(),
            self.data(),
            self.grad()
        )
    }
}

impl Add for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let out = Tensor::new(self.data() + rhs.data());
        out.0.borrow_mut()._prev = vec![self.clone(), rhs.clone()];
        out
    }
}

impl Mul for Tensor {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let out = Tensor::new(self.data() * rhs.data());
        out.0.borrow_mut()._prev = vec![self.clone(), rhs.clone()];
        out
    }
}
