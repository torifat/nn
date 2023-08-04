use ndarray::{arr2, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

// TODO: Try other activation functions like ReLU, Leaky ReLU, SiLU, Softmax, Hyperbolic Tangent, etc.
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

struct NeuralNetwork {
    weights1: Array2<f64>,
    weights2: Array2<f64>,
    bias1: Array2<f64>,
    bias2: Array2<f64>,
    hidden: Array2<f64>,
}

impl NeuralNetwork {
    fn new() -> Self {
        // TODO: Try more advanced weight initialization strategies like Xavier/Glorot or He initialization
        let weights1 = Array2::random((2, 4), Uniform::new(-1.0, 1.0));
        let weights2 = Array2::random((4, 1), Uniform::new(-1.0, 1.0));

        let bias1 = Array2::random((1, 4), Uniform::new(-1.0, 1.0));
        let bias2 = Array2::random((1, 1), Uniform::new(-1.0, 1.0));

        let hidden = Array2::zeros((1, 4));

        Self {
            weights1,
            weights2,
            bias1,
            bias2,
            hidden,
        }
    }

    fn sigmoid(&self, x: &Array2<f64>) -> Array2<f64> {
        x.mapv(sigmoid)
    }

    fn sigmoid_derivative(&self, x: &Array2<f64>) -> Array2<f64> {
        x * (1.0 - x)
    }

    fn feedforward(&mut self, inputs: &Array2<f64>) -> Array2<f64> {
        self.hidden = self.sigmoid(&(inputs.dot(&self.weights1) + &self.bias1));
        self.sigmoid(&(self.hidden.dot(&self.weights2) + &self.bias2))
    }

    fn train(&mut self, inputs: &Array2<f64>, targets: &Array2<f64>, iterations: usize) {
        for _ in 0..iterations {
            let outputs = self.feedforward(inputs);

            // Backpropagation
            let output_errors = targets - &outputs;
            let output_gradients = self.sigmoid_derivative(&outputs);
            let output_deltas = output_errors * output_gradients;

            let hidden_errors = output_deltas.dot(&self.weights2.t());
            let hidden_gradients = self.sigmoid_derivative(&self.hidden);
            let hidden_deltas = hidden_errors * hidden_gradients;

            // Update weights
            self.weights2
                .scaled_add(1.0, &self.hidden.t().dot(&output_deltas));
            self.weights1
                .scaled_add(1.0, &inputs.t().dot(&hidden_deltas));

            // Update biases
            self.bias2
                .scaled_add(1.0, &output_deltas.sum_axis(ndarray::Axis(0)));
            self.bias1
                .scaled_add(1.0, &hidden_deltas.sum_axis(ndarray::Axis(0)));
        }
    }
}

fn main() {
    let mut nn = NeuralNetwork::new();

    // Training data
    let inputs = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let targets = arr2(&[[0.0], [1.0], [1.0], [0.0]]);

    nn.train(&inputs, &targets, 10000);

    let inputs = arr2(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]);
    let outputs = nn.feedforward(&inputs);

    let outputs = outputs.mapv(|x| x.round() as i32);
    println!("Output:\n{:?}", outputs);
}
