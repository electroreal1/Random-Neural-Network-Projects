package neural_network.master_network;

import java.util.Random;

public class Neuron {
    private double[] weights;
    private double bias;
    private double output;
    private double delta; // for backpropagation
    private static final Random random = new Random();

    public Neuron(int inputSize) {
        weights = new double[inputSize];
        for (int i = 0; i < inputSize; i++) {
            weights[i] = random.nextGaussian() * 0.5;
        }

        bias = random.nextGaussian() * 0.5;
    }

    public double activate(double[] input) {
        if (input.length != weights.length) {
            throw new IllegalArgumentException("Input length not equal to weights length");
        }

        double sum = bias;
        for (int i = 0; i < input.length; i++) {
            sum += weights[i] * input[i];
        }

        output = sigmoid(sum);
        return output;
    }

    private double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public double sigmoidDerivative(double x) {
        return x * (1.0 - x);
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeight(int index, double weight) {
        weights[index] = weight;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getOutput() {
        return output;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }
}