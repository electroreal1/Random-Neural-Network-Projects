package neural_network.master_network;

public class Layer {
    private Neuron[] neurons;
    private int size;

    public Layer(int neuronCount, int inputSize) {
        this.size = neuronCount;
        this.neurons = new Neuron[neuronCount];

        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(inputSize);
        }
    }

    public double[] feedForward(double[] inputs) {
        double[] outputs = new double[size];
        for (int i = 0; i < size; i++) {
            outputs[i] = neurons[i].activate(inputs);
        }

        return outputs;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }

    public int getSize() {
        return size;
    }
}