package neural_network.games;

import neural_network.master_network.BrainManager;
import neural_network.Main;
import neural_network.master_network.MoodController;
import neural_network.master_network.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class SnakeGame extends JPanel implements Runnable {
    private volatile boolean active = true;
    private final Main hub;

    private final int POPULATION_SIZE = 100;
    private ArrayList<SnakeAI> population;
    private int generation = 1;
    private int maxScoreEver = 0;
    private SnakeAI bestAIEver;

    private static final int GRID_SIZE = 20;
    private static final int TILE_SIZE = 25;
    private Point food;
    private final Random random = new Random();

    private static class SnakeAI {
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
            return new SnakeAI(BrainManager.cloneAndMutate(this.brain));
        }
    }

    public SnakeGame(Main hub) {
        this.hub = hub;
        setBackground(Color.DARK_GRAY);
        setFocusable(true);
        initPopulation();
        spawnFood();
    }

    public void setActive(boolean active) {
        this.active = active;
        if (active && hub != null && !population.isEmpty()) {
            BrainManager.copyWeights(hub.getBrainManager().getMasterNetwork(), population.get(0).brain);
        }
    }

    private void initPopulation() {
        population = new ArrayList<>();
        for (int i = 0; i < POPULATION_SIZE; i++) {
            NeuralNetwork net = BrainManager.createNewNetwork();
            if (hub != null && i == 0) {
                BrainManager.copyWeights(hub.getBrainManager().getMasterNetwork(), net);
            }
            population.add(new SnakeAI(net));
        }
        bestAIEver = population.get(0);
    }

    public void startGame() {
        Thread gameThread = new Thread(this);
        gameThread.start();
    }

    private void spawnFood() {
        food = new Point(random.nextInt(GRID_SIZE), random.nextInt(GRID_SIZE));
    }

    @Override
    public void run() {
        boolean isRunning = true;
        while (isRunning) {
            if (!active) {
                try { Thread.sleep(100); } catch (InterruptedException e) { break; }
                continue;
            }
            updateGame();
            repaint();
            try { Thread.sleep(50); } catch (InterruptedException e) { break; }
        }
    }

    private void updateGame() {
        boolean allDead = true;

        for (SnakeAI ai : population) {
            if (!ai.isAlive) continue;
            allDead = false;

            Point head = ai.body.get(0);
            double[] inputs = new double[BrainManager.INPUT_SIZE];
            inputs[0] = head.x / (double) GRID_SIZE;
            inputs[1] = head.y / (double) GRID_SIZE;
            inputs[2] = food.x / (double) GRID_SIZE;
            inputs[3] = food.y / (double) GRID_SIZE;
            inputs[4] = head.x == 0 ? 1 : 0;
            inputs[5] = head.x == GRID_SIZE - 1 ? 1 : 0;
            inputs[6] = head.y == 0 ? 1 : 0;
            inputs[7] = head.y == GRID_SIZE - 1 ? 1 : 0;
            inputs[8] = ai.dir == 0 ? 1 : 0;
            inputs[9] = ai.dir == 1 ? 1 : 0;
            inputs[10] = ai.dir == 2 ? 1 : 0;
            inputs[11] = ai.dir == 3 ? 1 : 0;

            if (hub != null) {
                hub.getMoodController().injectMetaInputs(inputs, MoodController.GAME_SNAKE);
            }

            double[] outputs = ai.brain.predict(inputs);
            int bestDir = 0;
            double max = outputs[0];
            for (int i = 1; i < 4; i++) {
                if (outputs[i] > max) {
                    max = outputs[i];
                    bestDir = i;
                }
            }

            if (hub != null && ai == population.get(0)) {
                hub.getBrainManager().recordLiveState(ai.brain, inputs);
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
            int genBest = population.stream().mapToInt(a -> a.score).max().orElse(0);
            evolveNextGen();
            generation++;
            spawnFood();
            if (hub != null) {
                hub.onGenerationComplete(MoodController.GAME_SNAKE, genBest, maxScoreEver, bestAIEver.brain);
            }
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
        int offsetX = 75;
        int offsetY = 40;

        g.setColor(new Color(45, 45, 55));
        g.fillRect(offsetX, offsetY, GRID_SIZE * TILE_SIZE, GRID_SIZE * TILE_SIZE);

        g.setColor(Color.RED);
        g.fillRect(offsetX + food.x * TILE_SIZE, offsetY + food.y * TILE_SIZE, TILE_SIZE, TILE_SIZE);

        g.setColor(Color.GREEN);
        for (SnakeAI ai : population) {
            if (ai.isAlive) {
                for (Point p : ai.body) {
                    g.fillRect(offsetX + p.x * TILE_SIZE, offsetY + p.y * TILE_SIZE, TILE_SIZE - 1, TILE_SIZE - 1);
                }
                break;
            }
        }

        g.setColor(Color.WHITE);
        g.setFont(new Font("SansSerif", Font.BOLD, 16));
        g.drawString("Generation: " + generation + "   |   Max Score: " + maxScoreEver, offsetX, 28);
    }
}