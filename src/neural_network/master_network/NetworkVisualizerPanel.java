package neural_network.master_network;

import javax.swing.*;
import java.awt.*;
import java.util.List;

public class NetworkVisualizerPanel extends JPanel {
    private final BrainManager brainManager;

    public NetworkVisualizerPanel(BrainManager brainManager) {
        this.brainManager = brainManager;
        this.setPreferredSize(new Dimension(270, 790));
        this.setBackground(new Color(16, 18, 26));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        g2.setColor(Color.WHITE);
        g2.setFont(new Font("SansSerif", Font.BOLD, 14));
        g2.drawString("LIVE NEURAL TOPOLOGY", 20, 28);

        g2.setFont(new Font("SansSerif", Font.PLAIN, 11));
        g2.setColor(new Color(160, 175, 200));
        g2.drawString("Cyan = +Weight  |  Red = -Weight", 20, 46);

        NeuralNetwork net = brainManager.getDisplayedNetwork();
        double[] lastInputs = brainManager.getLastInputs();
        if (net == null || net.getLayers().size() < 2) return;

        List<Layer> layers = net.getLayers();
        Neuron[] hiddenNeurons = layers.get(0).getNeurons();
        Neuron[] outputNeurons = layers.get(1).getNeurons();

        int[] inputIndices = {0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 66, 67, 68, 69, 70, 71};
        int hiddenCount = Math.min(20, hiddenNeurons.length);
        int[] outputIndices = {0, 1, 2, 3, 4, 5, 6, 7, 64, 65, 66, 67};

        int topY = 75;
        int bottomY = getHeight() - 40;
        int usableHeight = bottomY - topY;

        int xInput = 35;
        int xHidden = 135;
        int xOutput = 230;

        Point[] inPts = new Point[inputIndices.length];
        Point[] hidPts = new Point[hiddenCount];
        Point[] outPts = new Point[outputIndices.length];

        for (int i = 0; i < inputIndices.length; i++) {
            int y = topY + (i * usableHeight) / (inputIndices.length - 1);
            inPts[i] = new Point(xInput, y);
        }
        for (int i = 0; i < hiddenCount; i++) {
            int y = topY + (i * usableHeight) / (hiddenCount - 1);
            hidPts[i] = new Point(xHidden, y);
        }
        for (int i = 0; i < outputIndices.length; i++) {
            int y = topY + (i * usableHeight) / (outputIndices.length - 1);
            outPts[i] = new Point(xOutput, y);
        }

        drawInputToHiddenSynapses(hiddenCount, hiddenNeurons, inputIndices, lastInputs, g2, inPts, hidPts);

        drawHiddenToOutputSynapses(outputIndices, outputNeurons, hiddenCount, hiddenNeurons, g2, hidPts, outPts);

        drawInputNodes(inputIndices, lastInputs, g2, inPts);

        drawHiddenNodes(hiddenCount, hiddenNeurons, g2, hidPts);

        drawOutputNodes(g2, outputIndices, outputNeurons, outPts);

        g2.setFont(new Font("SansSerif", Font.BOLD, 11));
        g2.setColor(new Color(150, 165, 190));
        g2.drawString("Inputs", xInput - 15, getHeight() - 12);
        g2.drawString("Hidden", xHidden - 18, getHeight() - 12);
        g2.drawString("Outputs", xOutput - 18, getHeight() - 12);
    }

    private static void drawInputToHiddenSynapses(int hiddenCount, Neuron[] hiddenNeurons, int[] inputIndices, double[] lastInputs, Graphics2D g2, Point[] inPts, Point[] hidPts) {
        for (int h = 0; h < hiddenCount; h++) {
            double[] weights = hiddenNeurons[h].getWeights();
            double hOut = hiddenNeurons[h].getOutput();
            for (int i = 0; i < inputIndices.length; i++) {
                int inIdx = inputIndices[i];
                if (inIdx >= weights.length) continue;

                double w = weights[inIdx];
                double signal = Math.abs(w * lastInputs[inIdx] * hOut);
                if (signal > 0.08) {
                    int alpha = (int) Math.min(210, signal * 220);
                    g2.setColor(w >= 0 ? new Color(0, 210, 255, alpha) : new Color(255, 60, 110, alpha));
                    g2.setStroke(new BasicStroke((float) Math.min(2.2f, 0.5f + signal)));
                    g2.drawLine(inPts[i].x, inPts[i].y, hidPts[h].x, hidPts[h].y);
                }
            }
        }
    }

    private static void drawHiddenToOutputSynapses(int[] outputIndices, Neuron[] outputNeurons, int hiddenCount, Neuron[] hiddenNeurons, Graphics2D g2, Point[] hidPts, Point[] outPts) {
        for (int o = 0; o < outputIndices.length; o++) {
            int outIdx = outputIndices[o];
            if (outIdx >= outputNeurons.length) continue;

            double[] weights = outputNeurons[outIdx].getWeights();
            double oOut = outputNeurons[outIdx].getOutput();
            for (int h = 0; h < hiddenCount; h++) {
                double w = weights[h];
                double signal = Math.abs(w * hiddenNeurons[h].getOutput() * oOut);
                if (signal > 0.08) {
                    int alpha = (int) Math.min(220, signal * 240);
                    g2.setColor(w >= 0 ? new Color(0, 230, 255, alpha) : new Color(255, 70, 120, alpha));
                    g2.setStroke(new BasicStroke((float) Math.min(2.4f, 0.6f + signal)));
                    g2.drawLine(hidPts[h].x, hidPts[h].y, outPts[o].x, outPts[o].y);
                }
            }
        }
    }

    private void drawInputNodes(int[] inputIndices, double[] lastInputs, Graphics2D g2, Point[] inPts) {
        for (int i = 0; i < inputIndices.length; i++) {
            double val = Math.max(0.0, Math.min(1.0, Math.abs(lastInputs[inputIndices[i]])));
            boolean isMoodInput = inputIndices[i] >= 64;
            Color baseColor = isMoodInput ? new Color(168, 85, 247) : new Color(59, 130, 246);
            drawNeuronNode(g2, inPts[i].x, inPts[i].y, 6, val, baseColor);
        }
    }

    private void drawHiddenNodes(int hiddenCount, Neuron[] hiddenNeurons, Graphics2D g2, Point[] hidPts) {
        for (int h = 0; h < hiddenCount; h++) {
            double val = hiddenNeurons[h].getOutput();
            drawNeuronNode(g2, hidPts[h].x, hidPts[h].y, 6, val, new Color(16, 185, 129));
        }
    }

    private void drawOutputNodes(Graphics2D g2, int[] outputIndices, Neuron[] outputNeurons, Point[] outPts) {
        String[] outLabels = {"A0", "A1", "A2", "A3", "A4", "A5", "A6", "A7", "Flp", "Snk", "Oth", "Swt"};
        g2.setFont(new Font("SansSerif", Font.BOLD, 9));
        for (int o = 0; o < outputIndices.length; o++) {
            int outIdx = outputIndices[o];
            double val = outputNeurons[outIdx].getOutput();
            boolean isDecisionNode = outIdx >= 64;
            Color baseColor = isDecisionNode ? new Color(245, 158, 11) : new Color(0, 220, 255);
            drawNeuronNode(g2, outPts[o].x, outPts[o].y, 7, val, baseColor);

            g2.setColor(new Color(200, 210, 230));
            g2.drawString(outLabels[o], outPts[o].x + 10, outPts[o].y + 3);
        }
    }

    private void drawNeuronNode(Graphics2D g2, int x, int y, int radius, double activation, Color activeColor) {
        g2.setColor(new Color(30, 35, 50));
        g2.fillOval(x - radius, y - radius, radius * 2, radius * 2);

        int glowAlpha = (int) Math.max(45, Math.min(255, activation * 255));
        g2.setColor(new Color(activeColor.getRed(), activeColor.getGreen(), activeColor.getBlue(), glowAlpha));
        g2.fillOval(x - radius, y - radius, radius * 2, radius * 2);

        g2.setColor(new Color(220, 230, 255, 180));
        g2.setStroke(new BasicStroke(1f));
        g2.drawOval(x - radius, y - radius, radius * 2, radius * 2);
    }
}
