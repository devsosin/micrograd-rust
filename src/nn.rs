use std::{cell::RefCell, hash::Hash, iter::zip, rc::Rc};

use rand::Rng;

use crate::engine::Tensor;

pub struct NeuronData {
    weights: Vec<Tensor>,
    bias: Tensor,
}

#[derive(Clone)]
pub struct Neuron(Rc<RefCell<NeuronData>>);

impl Neuron {
    pub fn new(n: usize) -> Self {
        let mut rng = rand::rng();
        // let mut weights = Vec::with_capacity(n);
        // for _ in 0..n {
        //     weights.push(Tensor::new(rng.random_range(-1.0..1.0), ""));
        // }
        let weights = (0..n)
            .map(|_| Tensor::new(rng.random_range(-1.0..1.0), ""))
            .collect();

        Self(Rc::new(RefCell::new(NeuronData {
            weights,
            bias: Tensor::new(rng.random_range(-1.0..1.0), ""),
        })))
    }

    pub fn weights(&self) -> Vec<Tensor> {
        self.0.borrow().weights.clone()
    }

    pub fn bias(&self) -> Tensor {
        self.0.borrow().bias.clone()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        [self.weights(), vec![self.bias()]].concat()
    }

    // __call__ method does not exists
    pub fn forward(&self, x: &Vec<Tensor>) -> Tensor {
        // do not need to make a tensor by default
        // because, it already implements Add<f64> with Tensor
        // and, it makes additional calculation fee.
        // but, when we use the multiple layer, the input x would be tensors.
        let data = zip(self.weights(), x)
            .map(|(w, x)| w * x)
            .fold(Tensor::new(0.0, "temp"), |acc, wx| acc + wx)
            + self.bias();

        data.tanh()
    }
}

impl PartialEq for Neuron {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for Neuron {}

impl Hash for Neuron {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_ptr().hash(state);
    }
}

pub struct LayerData {
    neurons: Vec<Neuron>,
}

#[derive(Clone)]
pub struct Layer(Rc<RefCell<LayerData>>);

impl Layer {
    pub fn new(n_in: usize, n_out: usize) -> Self {
        let neurons = (0..n_out).map(|_| Neuron::new(n_in)).collect();
        Self(Rc::new(RefCell::new(LayerData { neurons })))
    }

    pub fn neurons(&self) -> Vec<Neuron> {
        self.0.borrow().neurons.clone()
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.neurons()
            .iter()
            .map(|n| n.parameters())
            .flatten()
            .collect()
    }

    pub fn forward(&self, x: &Vec<Tensor>) -> Vec<Tensor> {
        self.neurons().iter().map(|n| n.forward(&x)).collect()
    }
}

pub struct MLPData {
    layers: Vec<Layer>,
}

#[derive(Clone)]
pub struct MLP(Rc<RefCell<MLPData>>);

impl MLP {
    // Vec<Sizes>
    pub fn new(n_in: usize, n_outs: Vec<usize>) -> Self {
        // first input, next output
        let nodes = [vec![n_in], n_outs].concat();
        let mut layers = Vec::new();
        for i in 0..nodes.len() - 1 {
            layers.push(Layer::new(nodes[i], nodes[i + 1]));
        }

        Self(Rc::new(RefCell::new(MLPData { layers })))
    }

    pub fn layers(&self) -> Vec<Layer> {
        self.0.borrow().layers.clone()
    }

    pub fn parameters(&self) -> Vec<Vec<Tensor>> {
        self.layers().iter().map(|l| l.parameters()).collect()
    }

    pub fn forward(&self, x: &Vec<Tensor>) -> Vec<Tensor> {
        // for l in self.layers() {
        //     let x = l.forward(x);
        // }
        // backward가 되려나? 된다.
        self.layers()
            .iter()
            .fold(x.clone(), |acc, l| l.forward(&acc))
    }
}

pub trait Module {
    fn zero_grad(&self);
    // fn.update();
}

impl Module for MLP {
    fn zero_grad(&self) {
        let _ = self.layers().iter().map(|l| l.zero_grad());
    }
}
impl Module for Layer {
    fn zero_grad(&self) {
        let _ = self.neurons().iter().map(|n| n.zero_grad());
    }
}

impl Module for Neuron {
    fn zero_grad(&self) {
        let _ = self.parameters().iter().map(|p| p.set_grad(0.0));
    }
}
