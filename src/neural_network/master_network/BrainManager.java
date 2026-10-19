package neural_network.master_network;

import java.io.File;
import java.util.List;
import java.util.Random;

public class BrainManager {
    public static final int INPUT_SIZE = 83;
    public static final int HIDDEN_SIZE = 48;
    public static final int OUTPUT_SIZE = 69;

    public static final String WEIGHTS_FILE = "brain_weights.txt";

    private final NeuralNetwork masterNetwork;
    private volatile NeuralNetwork displayedNetwork;
    private volatile double[] lastInputs = new double[INPUT_SIZE];
    private static final Random random = new Random();

    public BrainManager() {
        this.masterNetwork = createNewNetwork();
        this.displayedNetwork = this.masterNetwork;
        loadMasterWeights();
        registerShutdownHook();
    }

    public static NeuralNetwork createNewNetwork() {
        NeuralNetwork net = new NeuralNetwork(0.05);
        net.addLayer(new Layer(HIDDEN_SIZE, INPUT_SIZE));
        net.addLayer(new Layer(OUTPUT_SIZE, HIDDEN_SIZE));

        for (Layer layer : net.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                double[] w = neuron.getWeights();
                for (int i = 0; i < w.length; i++) {
                    neuron.setWeight(i, random.nextGaussian() * 1.5);
                }
                neuron.setBias(random.nextGaussian() * 0.5);
            }
        }
        return net;
    }

    public static NeuralNetwork cloneAndMutate(NeuralNetwork source) {
        NeuralNetwork mutant = new NeuralNetwork(0.05);
        mutant.addLayer(new Layer(HIDDEN_SIZE, INPUT_SIZE));
        mutant.addLayer(new Layer(OUTPUT_SIZE, HIDDEN_SIZE));

        List<Layer> origLayers = source.getLayers();
        List<Layer> mutLayers = mutant.getLayers();

        for (int i = 0; i < origLayers.size(); i++) {
            Neuron[] origNeurons = origLayers.get(i).getNeurons();
            Neuron[] mutNeurons = mutLayers.get(i).getNeurons();

            for (int j = 0; j < origNeurons.length; j++) {
                double[] ow = origNeurons[j].getWeights();
                double[] mw = mutNeurons[j].getWeights();

                for (int w = 0; w < ow.length; w++) {
                    if (random.nextDouble() < 0.2) {
                        mw[w] = ow[w] + random.nextGaussian() * 0.4;
                    } else {
                        mw[w] = ow[w];
                    }
                }

                double biasMutation = (random.nextDouble() < 0.1) ? random.nextGaussian() * 0.3 : 0.0;
                mutNeurons[j].setBias(origNeurons[j].getBias() + biasMutation);
            }
        }
        return mutant;
    }

    public static void copyWeights(NeuralNetwork source, NeuralNetwork destination) {
        List<Layer> origLayers = source.getLayers();
        List<Layer> destLayers = destination.getLayers();

        for (int l = 0; l < origLayers.size(); l++) {
            Neuron[] origNeurons = origLayers.get(l).getNeurons();
            Neuron[] destNeurons = destLayers.get(l).getNeurons();

            for (int n = 0; n < origNeurons.length; n++) {
                double[] ow = origNeurons[n].getWeights();
                for (int w = 0; w < ow.length; w++) {
                    destNeurons[n].setWeight(w, ow[w]);
                }
                destNeurons[n].setBias(origNeurons[n].getBias());
            }
        }
    }

    public void recordLiveState(NeuralNetwork activeNet, double[] inputs) {
        this.displayedNetwork = activeNet;
        if (inputs != null && inputs.length == INPUT_SIZE) {
            this.lastInputs = inputs.clone();
        }
    }

    public NeuralNetwork getDisplayedNetwork() {
        return displayedNetwork != null ? displayedNetwork : masterNetwork;
    }

    public double[] getLastInputs() {
        return lastInputs;
    }

    public synchronized NeuralNetwork getMasterNetwork() {
        return masterNetwork;
    }

    public synchronized void updateMasterNetwork(NeuralNetwork bestGenerationBrain) {
        copyWeights(bestGenerationBrain, masterNetwork);
        this.displayedNetwork = masterNetwork;
    }

    public void loadMasterWeights() {
        File file = new File(WEIGHTS_FILE);
        if (file.exists()) {
            masterNetwork.loadWeights(WEIGHTS_FILE);
        }
    }

    private void registerShutdownHook() {
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            masterNetwork.saveWeights(WEIGHTS_FILE);
            System.out.println("Auto-saved Brain weights to " + WEIGHTS_FILE);
        }));
    }
}