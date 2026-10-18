package neural_network.games;

import neural_network.Layer;
import neural_network.NeuralNetwork;
import neural_network.Neuron;

import javax.swing.*;
import java.awt.*;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class FlappyBirdGame extends JPanel implements Runnable {
    private Thread gameThread;
    private boolean isRunning = true;

    private final int POPULATION_SIZE = 500;
    private ArrayList<Bird> birds;
    private int generation = 1;
    private int maxScoreEver = 0;
    private int currentScore = 0;
    private Bird bestBirdEver;

    private final String weightsFile = "flappy_weights.txt";

    private static class Pipe {
        int x, topHeight, gapSize;
        boolean passed = false;
        Pipe(int x, int topHeight, int gapSize) {
            this.x = x;
            this.topHeight = topHeight;
            this.gapSize = gapSize;
        }
    }

    private ArrayList<Pipe> pipes;
    private final int pipeWidth = 60;
    private final int pipeSpeed = 4;
    private final Random random = new Random();

    private class Bird {
        double y = 250;
        double velocity = 0;
        double fitness = 0;
        int score = 0;
        boolean isAlive = true;
        NeuralNetwork brain;

        Bird(NeuralNetwork brain) {
            this.brain = brain;
        }

        Bird cloneAndMutate() {
            NeuralNetwork mutantBrain = new NeuralNetwork(0.1);
            mutantBrain.addLayer(new Layer(8, 5));
            mutantBrain.addLayer(new Layer(1, 8));

            List<Layer> originalLayers = this.brain.getLayers();
            List<Layer> mutantLayers = mutantBrain.getLayers();

            for (int l = 0; l < originalLayers.size(); l++) {
                Neuron[] origNeurons = originalLayers.get(l).getNeurons();
                Neuron[] mutNeurons = mutantLayers.get(l).getNeurons();

                for (int n = 0; n < origNeurons.length; n++) {
                    double[] origWeights = origNeurons[n].getWeights();
                    double[] mutWeights = mutNeurons[n].getWeights();

                    for (int w = 0; w < origWeights.length; w++) {
                        if (random.nextDouble() < 0.2) {
                            mutWeights[w] = origWeights[w] + random.nextGaussian() * 0.5;
                        } else {
                            mutWeights[w] = origWeights[w];
                        }
                    }
                    mutNeurons[n].setBias(origNeurons[n].getBias() + (random.nextDouble() < 0.1 ? random.nextGaussian() * 0.3 : 0));
                }
            }
            return new Bird(mutantBrain);
        }
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Neural Flappy Bird Evolution Arena");
        FlappyBirdGame game = new FlappyBirdGame();
        frame.add(game);
        frame.setSize(480, 680);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        game.startGame();
    }

    public FlappyBirdGame() {
        setBackground(new Color(113, 197, 207));
        setFocusable(true);

        birds = new ArrayList<>();
        pipes = new ArrayList<>();

        initPopulation();

        File f = new File(weightsFile);
        if (f.exists() && !birds.isEmpty()) {
            birds.get(0).brain.loadWeights(weightsFile);
            bestBirdEver = birds.get(0);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (bestBirdEver != null) {
                bestBirdEver.brain.saveWeights(weightsFile);
                System.out.println("Auto-saved best Flappy Bird AI weights to " + weightsFile);
            }
        }));

        resetRound();
    }

    private void initPopulation() {
        birds.clear();
        for (int i = 0; i < POPULATION_SIZE; i++) {
            NeuralNetwork net = new NeuralNetwork(0.1);
            net.addLayer(new Layer(8, 5));
            net.addLayer(new Layer(1, 8));

            initializeRandomWeights(net);
            birds.add(new Bird(net));
        }
        bestBirdEver = birds.get(0);
    }

    private void initializeRandomWeights(NeuralNetwork net) {
        for (Layer layer : net.getLayers()) {
            for (Neuron neuron : layer.getNeurons()) {
                double[] weights = neuron.getWeights();
                for (int i = 0; i < weights.length; i++) {
                    neuron.setWeight(i, random.nextGaussian() * 2.0);
                }
                neuron.setBias(random.nextGaussian());
            }
        }
    }

    public void startGame() {
        gameThread = new Thread(this);
        gameThread.start();
    }

    private void resetRound() {
        pipes.clear();
        int startX = 480;
        for (int i = 0; i < 3; i++) {
            int topHeight = 50 + random.nextInt(280);
            int gapSize = 120 + random.nextInt(40);
            pipes.add(new Pipe(startX, topHeight, gapSize));
            startX += 240 + random.nextInt(60);
        }

        currentScore = 0;
        for (Bird b : birds) {
            b.y = 250;
            b.velocity = 0;
            b.score = 0;
            b.fitness = 0;
            b.isAlive = true;
        }
    }

    @Override
    public void run() {
        while (isRunning) {
            updateGame();
            repaint();
            try {
                Thread.sleep(20);
            } catch (InterruptedException e) {
                break;
            }
        }
    }

    private void updateGame() {
        double gravity = 0.5;
        double lift = -8.5;

        Pipe closestPipe = null;
        for (Pipe p : pipes) {
            if (p.x + pipeWidth > 80) {
                closestPipe = p;
                break;
            }
        }

        boolean allDead = true;

        for (Bird bird : birds) {
            if (!bird.isAlive) continue;
            allDead = false;

            bird.velocity += gravity;
            bird.y += bird.velocity;
            bird.fitness += 0.1;

            if (closestPipe != null) {
                double[] inputs = {
                        bird.y / 680.0,
                        (bird.velocity + 10) / 20.0,
                        (closestPipe.x - 80) / 480.0,
                        closestPipe.topHeight / 680.0,
                        (closestPipe.topHeight + closestPipe.gapSize) / 680.0
                };

                double[] output = bird.brain.predict(inputs);
                if (output[0] > 0.5) {
                    bird.velocity = lift;
                }
            }

            for (Pipe p : pipes) {
                if (80 + 30 > p.x && 80 < p.x + pipeWidth) {
                    if (bird.y < p.topHeight || bird.y + 30 > p.topHeight + p.gapSize) {
                        bird.isAlive = false;
                    }
                }
            }

            if (bird.y > 620 || bird.y < 0) {
                bird.isAlive = false;
            }
        }

        if (allDead) {
            evolveNextGeneration();
            generation++;
            resetRound();
            return;
        }

        for (Pipe p : pipes) {
            p.x -= pipeSpeed;
            if (!p.passed && p.x < 80) {
                p.passed = true;
                for (Bird b : birds) {
                    if (b.isAlive) {
                        b.score++;
                        b.fitness += 10;
                        if (b.score > currentScore) currentScore = b.score;
                        if (b.score > maxScoreEver) {
                            maxScoreEver = b.score;
                            bestBirdEver = b;
                        }
                    }
                }
            }
        }

        if (!pipes.isEmpty() && pipes.get(0).x < -pipeWidth) {
            pipes.remove(0);
            int lastX = pipes.get(pipes.size() - 1).x;

            int newTopHeight = 50 + random.nextInt(280);
            int newGapSize = 120 + random.nextInt(40);
            int randomSpacing = 220 + random.nextInt(80);

            pipes.add(new Pipe(lastX + randomSpacing, newTopHeight, newGapSize));
        }
    }

    private void evolveNextGeneration() {
        Bird bestBird = birds.get(0);
        for (Bird b : birds) {
            if (b.fitness > bestBird.fitness) {
                bestBird = b;
            }
        }

        if (bestBird.score >= bestBirdEver.score) {
            bestBirdEver = bestBird;
        }

        ArrayList<Bird> nextGen = new ArrayList<>();
        nextGen.add(new Bird(bestBird.brain));

        while (nextGen.size() < POPULATION_SIZE) {
            nextGen.add(bestBird.cloneAndMutate());
        }

        birds = nextGen;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);

        g.setColor(new Color(117, 190, 36));
        for (Pipe p : pipes) {
            g.fillRect(p.x, 0, pipeWidth, p.topHeight);
            g.fillRect(p.x, p.topHeight + p.gapSize, pipeWidth, 680 - (p.topHeight + p.gapSize));
        }

        for (Bird b : birds) {
            if (b.isAlive) {
                g.setColor(new Color(248, 222, 34, 180));
                g.fillOval(80, (int) b.y, 30, 30);
                g.setColor(Color.BLACK);
                g.drawOval(80, (int) b.y, 30, 30);
            }
        }

        g.setFont(new Font("SansSerif", Font.BOLD, 18));
        g.setColor(Color.WHITE);
        g.drawString("Generation: " + generation, 20, 30);
        g.drawString("Max Score Ever: " + maxScoreEver, 20, 55);
        g.drawString("Current Score: " + currentScore, 20, 80);

        long aliveCount = birds.stream().filter(b -> b.isAlive).count();
        g.drawString("Birds Alive: " + aliveCount + " / " + POPULATION_SIZE, 20, 105);
    }
}