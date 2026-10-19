package neural_network.games;

import neural_network.master_network.BrainManager;
import neural_network.Main;
import neural_network.master_network.MoodController;
import neural_network.master_network.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.Random;

public class FlappyBirdGame extends JPanel implements Runnable {
    private boolean isRunning = true;
    private volatile boolean active = true;
    private final Main hub;

    private final int POPULATION_SIZE = 500;
    private ArrayList<Bird> birds;
    private int generation = 1;
    private int maxScoreEver = 0;
    private int currentScore = 0;
    private Bird bestBirdEver;

    private static class Pipe {
        int x, topHeight, gapSize;
        boolean passed = false;
        Pipe(int x, int topHeight, int gapSize) {
            this.x = x;
            this.topHeight = topHeight;
            this.gapSize = gapSize;
        }
    }

    private final ArrayList<Pipe> pipes;
    private final int pipeWidth = 60;
    private final int pipeSpeed = 4;
    private final Random random = new Random();

    private static class Bird {
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
            return new Bird(BrainManager.cloneAndMutate(this.brain));
        }
    }

    public FlappyBirdGame(Main hub) {
        this.hub = hub;
        this.setBackground(new Color(113, 197, 207));
        this.setFocusable(true);

        this.birds = new ArrayList<>();
        this.pipes = new ArrayList<>();

        this.initPopulation();
        this.resetRound();
    }

    public void setActive(boolean active) {
        this.active = active;
        if (active && hub != null && !birds.isEmpty()) {
            BrainManager.copyWeights(hub.getBrainManager().getMasterNetwork(), birds.get(0).brain);
        }
    }

    private void initPopulation() {
        birds.clear();
        for (int i = 0; i < POPULATION_SIZE; i++) {
            NeuralNetwork net = BrainManager.createNewNetwork();
            if (hub != null && i == 0) {
                BrainManager.copyWeights(hub.getBrainManager().getMasterNetwork(), net);
            }
            birds.add(new Bird(net));
        }
        bestBirdEver = birds.get(0);
    }

    public void startGame() {
        Thread gameThread = new Thread(this);
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
            if (!active) {
                try { Thread.sleep(100); } catch (InterruptedException e) { break; }
                continue;
            }
            updateGame();
            repaint();
            try { Thread.sleep(20); } catch (InterruptedException e) { break; }
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
                double[] inputs = new double[BrainManager.INPUT_SIZE];
                inputs[0] = bird.y / 680.0;
                inputs[1] = (bird.velocity + 10) / 20.0;
                inputs[2] = (closestPipe.x - 80) / 480.0;
                inputs[3] = closestPipe.topHeight / 680.0;
                inputs[4] = (closestPipe.topHeight + closestPipe.gapSize) / 680.0;

                if (hub != null) hub.getMoodController().injectMetaInputs(inputs, MoodController.GAME_FLAPPY);


                double[] output = bird.brain.predict(inputs);
                if (output[0] > 0.5) bird.velocity = lift;

                if (hub != null && bird == birds.get(0)) {
                    hub.getBrainManager().recordLiveState(bird.brain, inputs);
                }
                if (output[0] > 0.5) bird.velocity = lift;
            }

            for (Pipe p : pipes) {
                if (80 + 30 > p.x && 80 < p.x + pipeWidth) {
                    if (bird.y < p.topHeight || bird.y + 30 > p.topHeight + p.gapSize) {
                        bird.isAlive = false;
                    }
                }
            }

            if (bird.y > 600 || bird.y < 0) bird.isAlive = false;
        }



        if (allDead) {
            int genBestScore = currentScore;
            evolveNextGeneration();
            generation++;
            resetRound();
            if (hub != null) {
                hub.onGenerationComplete(MoodController.GAME_FLAPPY, genBestScore, maxScoreEver, bestBirdEver.brain);
            }
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
            if (b.fitness > bestBird.fitness) bestBird = b;
        }

        if (bestBird.score >= bestBirdEver.score) bestBirdEver = bestBird;


        ArrayList<Bird> nextGen = new ArrayList<>();
        nextGen.add(new Bird(bestBird.brain));

        while (nextGen.size() < POPULATION_SIZE) nextGen.add(bestBird.cloneAndMutate());
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