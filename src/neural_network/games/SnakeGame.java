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

public class SnakeGame extends JPanel implements Runnable {
    private Thread gameThread;
    private boolean isRunning = true;

    private final int POPULATION_SIZE = 100;
    private ArrayList<SnakeAI> population;
    private int generation = 1;
    private int maxScoreEver = 0;
    private SnakeAI bestAIEver;

    private final String weightsFile = "snake_weights.txt";
    private static final int GRID_SIZE = 20;
    private static final int TILE_SIZE = 25;
    private Point food;
    private Random random = new Random();

    private class SnakeAI {
        NeuralNetwork brain;
        double fitness = 0;
        int score = 0;
        boolean isAlive = true;
        List<Point> body = new ArrayList<>();
        int dir = 0;
        int stepsWithoutFood = 0;

        SnakeAI(NeuralNetwork brain) {
            this.brain = brain;
            reset();
        }

        void reset() {
            body.clear();
            body.add(new Point(10, 10));
            body.add(new Point(10, 11));
            body.add(new Point(10, 12));
            dir = 0;
            fitness = 0;
            score = 0;
            stepsWithoutFood = 0;
            isAlive = true;
        }

        SnakeAI cloneAndMutate() {
            NeuralNetwork mutant = new NeuralNetwork(0.1);
            mutant.addLayer(new Layer(16, 12));
            mutant.addLayer(new Layer(4, 16));

            List<Layer> origL = this.brain.getLayers();
            List<Layer> mutL = mutant.getLayers();

            for (int l = 0; l < origL.size(); l++) {
                Neuron[] origN = origL.get(l).getNeurons();
                Neuron[] mutN = mutL.get(l).getNeurons();
                for (int n = 0; n < origN.length; n++) {
                    double[] ow = origN[n].getWeights();
                    double[] mw = mutN[n].getWeights();
                    for (int w = 0; w < ow.length; w++) {
                        mw[w] = ow[w] + (random.nextDouble() < 0.2 ? random.nextGaussian() * 0.5 : 0);
                    }
                    mutN[n].setBias(origN[n].getBias() + (random.nextDouble() < 0.1 ? random.nextGaussian() * 0.3 : 0));
                }
            }
            return new SnakeAI(mutant);
        }
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Neural Snake Evolution Arena");
        SnakeGame game = new SnakeGame();
        frame.add(game);
        frame.setSize(530, 570);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        game.startGame();
    }

    public SnakeGame() {
        setBackground(Color.DARK_GRAY);
        setFocusable(true);
        initPopulation();

        File f = new File(weightsFile);
        if (f.exists() && !population.isEmpty()) {
            population.get(0).brain.loadWeights(weightsFile);
            bestAIEver = population.get(0);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (bestAIEver != null) {
                bestAIEver.brain.saveWeights(weightsFile);
                System.out.println("Auto-saved Snake AI weights to " + weightsFile);
            }
        }));

        spawnFood();
    }

    private void initPopulation() {
        population = new ArrayList<>();
        for (int i = 0; i < POPULATION_SIZE; i++) {
            NeuralNetwork net = new NeuralNetwork(0.1);
            net.addLayer(new Layer(16, 12));
            net.addLayer(new Layer(4, 16));
            for (Layer layer : net.getLayers()) {
                for (Neuron neuron : layer.getNeurons()) {
                    double[] w = neuron.getWeights();
                    for (int j = 0; j < w.length; j++) neuron.setWeight(j, random.nextGaussian() * 2.0);
                    neuron.setBias(random.nextGaussian());
                }
            }
            population.add(new SnakeAI(net));
        }
        bestAIEver = population.get(0);
    }

    public void startGame() {
        gameThread = new Thread(this);
        gameThread.start();
    }

    private void spawnFood() {
        food = new Point(random.nextInt(GRID_SIZE), random.nextInt(GRID_SIZE));
    }

    @Override
    public void run() {
        while (isRunning) {
            updateGame();
            repaint();
            try {
                Thread.sleep(80);
            } catch (InterruptedException e) {
                break;
            }
        }
    }

    private void updateGame() {
        boolean allDead = true;

        for (SnakeAI ai : population) {
            if (!ai.isAlive) continue;
            allDead = false;

            Point head = ai.body.get(0);
            double[] inputs = {
                    head.x / (double) GRID_SIZE,
                    head.y / (double) GRID_SIZE,
                    food.x / (double) GRID_SIZE,
                    food.y / (double) GRID_SIZE,
                    head.x == 0 ? 1 : 0,
                    head.x == GRID_SIZE - 1 ? 1 : 0,
                    head.y == 0 ? 1 : 0,
                    head.y == GRID_SIZE - 1 ? 1 : 0,
                    ai.dir == 0 ? 1 : 0,
                    ai.dir == 1 ? 1 : 0,
                    ai.dir == 2 ? 1 : 0,
                    ai.dir == 3 ? 1 : 0
            };

            double[] outputs = ai.brain.predict(inputs);
            int bestDir = 0;
            double max = outputs[0];
            for (int i = 1; i < 4; i++) {
                if (outputs[i] > max) {
                    max = outputs[i];
                    bestDir = i;
                }
            }

            if (!(ai.dir == 0 && bestDir == 2) && !(ai.dir == 2 && bestDir == 0) &&
                    !(ai.dir == 1 && bestDir == 3) && !(ai.dir == 3 && bestDir == 1)) {
                ai.dir = bestDir;
            }

            Point newHead = new Point(head);
            if (ai.dir == 0) newHead.y--;
            if (ai.dir == 1) newHead.x++;
            if (ai.dir == 2) newHead.y++;
            if (ai.dir == 3) newHead.x--;

            if (newHead.x < 0 || newHead.x >= GRID_SIZE || newHead.y < 0 || newHead.y >= GRID_SIZE) {
                ai.isAlive = false;
                continue;
            }
            for (Point p : ai.body) {
                if (p.equals(newHead)) {
                    ai.isAlive = false;
                    break;
                }
            }
            if (!ai.isAlive) continue;

            ai.body.add(0, newHead);
            ai.fitness += 0.1;
            ai.stepsWithoutFood++;

            if (newHead.equals(food)) {
                ai.score++;
                ai.fitness += 100;
                ai.stepsWithoutFood = 0;
                spawnFood();
                if (ai.score > maxScoreEver) {
                    maxScoreEver = ai.score;
                    bestAIEver = ai;
                }
            } else {
                ai.body.remove(ai.body.size() - 1);
            }

            if (ai.stepsWithoutFood > 200) {
                ai.isAlive = false;
            }
        }

        if (allDead) {
            evolveNextGen();
            generation++;
            spawnFood();
        }
    }

    private void evolveNextGen() {
        population.sort((a, b) -> Double.compare(b.fitness, a.fitness));
        if (population.get(0).score >= bestAIEver.score) {
            bestAIEver = population.get(0);
        }
        ArrayList<SnakeAI> nextGen = new ArrayList<>();
        nextGen.add(new SnakeAI(population.get(0).brain));
        while (nextGen.size() < POPULATION_SIZE) {
            nextGen.add(population.get(0).cloneAndMutate());
        }
        population = nextGen;
        for (SnakeAI ai : population) ai.reset();
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.setColor(Color.RED);
        g.fillRect(food.x * TILE_SIZE, food.y * TILE_SIZE, TILE_SIZE, TILE_SIZE);

        g.setColor(Color.GREEN);
        for (SnakeAI ai : population) {
            if (ai.isAlive) {
                for (Point p : ai.body) {
                    g.fillRect(p.x * TILE_SIZE, p.y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1);
                }
                break;
            }
        }

        g.setColor(Color.WHITE);
        g.setFont(new Font("SansSerif", Font.BOLD, 14));
        g.drawString("Generation: " + generation, 20, 25);
        g.drawString("Max Score: " + maxScoreEver, 20, 45);
    }
}