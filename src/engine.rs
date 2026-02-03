use std::{
    cell::RefCell,
    collections::HashSet,
    fmt::Debug,
    hash::Hash,
    ops::{Add, Mul},
    rc::Rc,
};

#[derive(PartialEq, Eq)]
enum Operation {
    None,
    Add,
    Mul,
    Tanh,
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
        // 기존 방식 -> 19개 중복 생김
        // hashing 할 때 참조하는 child가 복사해서 가져간 새 메모리여서 그런걸로 예상됨
        // -> todo에 push를 두번해서 중복 발생한 것
        let mut visited = HashSet::new();
        let mut todos = self._topological_sort(&mut visited);
        todos.reverse();

        // let mut todos = self.topological_sort();
        // todos.reverse();
        println!("todos: {:?}", todos.len());
        println!("todos: {:?}", todos);

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
                p.set_grad((1.0 - out.data().powi(2)) * out.grad());
            }
        }

        out.0.borrow_mut()._backward = Some(_backward);

        out
    }

    // temporal functions
    pub fn set_data(&self, data: f64) {
        self.0.borrow_mut().data = data;
    }

    // 기존 구현체
    fn _topological_sort(&self, visited: &mut HashSet<Tensor>) -> Vec<Tensor> {
        let mut todo: Vec<Tensor> = vec![];
        if !visited.contains(self) {
            visited.insert(self.clone());

            // 이부분에서 self를 빌린 hashing이 다르게 들어간건가?
            for p in &self.prev() {
                if !visited.contains(p) {
                    for child in &p._topological_sort(visited) {
                        todo.push(child.clone());
                    }
                    // 아 hashing 문제가 아니라 이부분 중복
                    // todo.push(p.clone());
                }
            }
            todo.push(self.clone());
        }

        todo
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

impl Add for &Tensor {
    type Output = Tensor;

    fn add(self, rhs: Self) -> Self::Output {
        let out = Tensor::new_with_operation(self.data() + rhs.data(), Operation::Add);
        out.0.borrow_mut()._prev = vec![self.clone(), rhs.clone()];

        fn _backward(out: &Tensor) {
            let prev = out.prev();
            for p in &prev {
                p.set_grad(1.0 * out.grad());
            }
        }

        out.0.borrow_mut()._backward = Some(_backward);

        out
    }
}

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
            l.set_grad(r.data() * out.grad());
            r.set_grad(l.data() * out.grad());
        }

        out.0.borrow_mut()._backward = Some(_backward);

        out
    }
}

pub trait Activation {
    fn tanh(&self) -> Tensor;
}
