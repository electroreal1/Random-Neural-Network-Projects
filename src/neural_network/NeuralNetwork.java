package neural_network;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetwork {
    private List<Layer> layers;
    private double learningRate;

    public NeuralNetwork(double learningRate) {
        this.learningRate = learningRate;
        this.layers = new ArrayList<>();
    }

    public void addLayer(Layer layer) {
        layers.add(layer);
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public double[] predict(double[] inputs) {
        double[] outputs = inputs;

        for (Layer layer : layers) {
            outputs = layer.feedForward(outputs);
        }

        return outputs;
    }

    public double trainSingleEpoch(double[][] trainingInputs, double[][] trainingOutputs) {
        double totalError = 0;

        for (int i = 0; i < trainingInputs.length; i++) {
            double[] predicted = predict(trainingInputs[i]);
            double[] errors = new double[predicted.length];

            for (int j = 0; j < predicted.length; j++) {
                errors[j] = trainingOutputs[i][j] - predicted[j];
                totalError += Math.pow(errors[j], 2);
            }

            backpropagate(trainingInputs[i], errors);
        }

        return totalError / (trainingInputs.length * trainingOutputs[0].length);
    }

    public void train(double[][] trainingInputs, double[][] trainingOutputs, int epochs) {
        for (int epoch = 0; epoch < epochs; epoch++) {
            double mse = trainSingleEpoch(trainingInputs, trainingOutputs);

            if (epoch % 2000 == 0) {
                System.out.println("Epoch " + epoch + ", Error: " + mse);
            }
        }
    }

    private void backpropagate(double[] inputs, double[] errors) {
        Layer outputLayer = layers.get(layers.size() - 1);
        Neuron[] outputNeurons = outputLayer.getNeurons();

        for (int i = 0; i < outputNeurons.length; i++) {
            double output = outputNeurons[i].getOutput();
            double delta = errors[i] * outputNeurons[i].sigmoidDerivative(output);
            outputNeurons[i].setDelta(delta);
        }

        for (int layerIndex = layers.size() - 2; layerIndex >= 0; layerIndex--) {
            Layer currentLayer = layers.get(layerIndex);
            Layer nextLayer = layers.get(layerIndex + 1);

            Neuron[] currentNeurons = currentLayer.getNeurons();
            Neuron[] nextNeurons = nextLayer.getNeurons();

            for (int i = 0; i < currentNeurons.length; i++) {
                double error = 0;

                for (int j = 0; j < nextNeurons.length; j++) {
                    error += nextNeurons[j].getDelta() * nextNeurons[j].getWeights()[i];
                }

                double output = currentNeurons[i].getOutput();
                double delta = error * currentNeurons[i].sigmoidDerivative(output);
                currentNeurons[i].setDelta(delta);
            }
        }

        updateWeights(inputs);
    }

    private void updateWeights(double[] inputs) {
        for (int layerIndex = 0; layerIndex < layers.size(); layerIndex++) {
            Layer layer = layers.get(layerIndex);

            double[] layerInputs;
            if (layerIndex == 0) {
                layerInputs = inputs;
            } else {
                Layer previousLayer = layers.get(layerIndex - 1);
                Neuron[] previousNeurons = previousLayer.getNeurons();
                layerInputs = new double[previousNeurons.length];
                for (int i = 0; i < previousNeurons.length; i++) {
                    layerInputs[i] = previousNeurons[i].getOutput();
                }
            }

            Neuron[] neurons = layer.getNeurons();
            for (Neuron neuron : neurons) {
                double[] weights = neuron.getWeights();
                for (int i = 0; i < weights.length; i++) {
                    double newWeight = weights[i] + learningRate * neuron.getDelta() * layerInputs[i];
                    neuron.setWeight(i, newWeight);
                }

                double newBias = neuron.getBias() + learningRate * neuron.getDelta();
                neuron.setBias(newBias);
            }
        }
    }

    public void saveWeights(String filename) {
        try (PrintWriter out = new PrintWriter(filename)) {
            for (Layer layer : layers) {
                out.println(layer.getSize());
                for (Neuron neuron : layer.getNeurons()) {
                    for (double weight : neuron.getWeights()) {
                        out.print(weight + ",");
                    }
                    out.println(neuron.getBias());
                }
            }
            System.out.println("Weights saved to " + filename);
        } catch (Exception e) {
            System.err.println("Failed to save weights: " + e.getMessage());
        }
    }

    public void loadWeights(String filename) {
        try (BufferedReader br = new BufferedReader(new FileReader(filename))) {
            for (Layer layer : layers) {
                br.readLine();
                for (Neuron neuron : layer.getNeurons()) {
                    String[] parts = br.readLine().split(",");
                    for (int i = 0; i < parts.length - 1; i++) {
                        neuron.setWeight(i, Double.parseDouble(parts[i]));
                    }
                    neuron.setBias(Double.parseDouble(parts[parts.length - 1]));
                }
            }
            System.out.println("Weights loaded from " + filename);
        } catch (Exception e) {
            System.err.println("Failed to load weights: " + e.getMessage());
        }
    }
}