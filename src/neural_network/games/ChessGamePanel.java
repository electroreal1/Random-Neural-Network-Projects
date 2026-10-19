package neural_network.games;

import neural_network.Main;
import neural_network.master_network.BrainManager;
import neural_network.master_network.NeuralNetwork;

import javax.swing.*;
import java.awt.*;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class ChessGamePanel extends JPanel implements Runnable {
    private volatile boolean active = false;
    private final Main hub;

    private final int POPULATION_SIZE = 25;
    private ArrayList<ChessAI> population;
    private int generation = 1;
    private int bestScoreEver = 0;

    // 8x8 Chess Board: positive = White/AI (1 to 100), negative = Black/Opponent (-1 to -100), 0 = Empty
    private final int[][] board = new int[8][8];
    private boolean aiTurn = true; // True = AI (White), False = Opponent (Black)
    private boolean playMode = false;
    private ChessAI currentBestAI;

    private static class ChessAI {
        NeuralNetwork brain;
        double fitness = 0;
        int score = 0;

        ChessAI(NeuralNetwork brain) {
            this.brain = brain;
        }

        ChessAI cloneAndMutate() {
            return new ChessAI(BrainManager.cloneAndMutate(this.brain));
        }
    }

    public ChessGamePanel(Main hub) {
        this.hub = hub;
        setBackground(new Color(110, 70, 40));
        setFocusable(true);

        initPopulation();

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                requestFocusInWindow();
                if (playMode && !aiTurn && !isGameOver()) {
                    int cellSize = 55;
                    int startX = 30;
                    int startY = 30;
                    int c = (e.getX() - startX) / cellSize;
                    int r = (e.getY() - startY) / cellSize;

                    if (r >= 0 && r < 8 && c >= 0 && c < 8) {
                        aiTurn = true;
                        repaint();
                    }
                }
            }
        });

        addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (e.getKeyCode() == KeyEvent.VK_P) {
                    playMode = !playMode;
                    resetBoard();
                }
            }
        });

        resetBoard();
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
            population.add(new ChessAI(net));
        }
        currentBestAI = population.get(0);
    }

    public void startGame() {
        Thread gameThread = new Thread(this);
        gameThread.start();
    }

    private void resetBoard() {
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                board[r][c] = 0;
            }
        }
        // Standard chess setup: Rook, Knight, Bishop, Queen, King, Bishop, Knight, Rook
        int[] backRow = {5, 3, 3, 9, 100, 3, 3, 5};
        for (int c = 0; c < 8; c++) {
            board[0][c] = -backRow[c]; // Black back rank
            board[1][c] = -1; // Black pawns
            board[6][c] = 1; // White pawns
            board[7][c] = backRow[c];  // White back rank
        }
        aiTurn = true;
    }

    @Override
    public void run() {
        boolean isRunning = true;
        while (isRunning) {
            if (!active) {
                try { Thread.sleep(100); } catch (InterruptedException e) { break; }
                continue;
            }

            if (!playMode) {
                playGeneration();
                try { Thread.sleep(250); } catch (InterruptedException e) { break; }
            } else {
                if (aiTurn && !isGameOver()) {
                    executeAIMove(currentBestAI);
                    aiTurn = false;
                    repaint();
                }
                try { Thread.sleep(800); } catch (InterruptedException e) { break; }
            }
        }
    }

    private void playGeneration() {
        int genBestScore = 0;
        for (int i = 0; i < population.size(); i++) {
            if (!active || playMode) return;

            ChessAI ai = population.get(i);
            boolean watchLive = (i == 0);
            resetBoard();
            boolean gameOver = false;
            int steps = 0;

            while (!gameOver && steps < 30 && active && !playMode) {
                executeAIMove(ai);
                ai.fitness += 2;
                steps++;

                if (watchLive) {
                    repaint();
                    try { Thread.sleep(120); } catch (InterruptedException e) { return; }
                }

                if (isGameOver() || steps >= 30) {
                    gameOver = true;
                    int score = evaluateBoardMaterial();
                    ai.fitness += score * 5;
                    ai.score = score;
                    if (score > genBestScore) genBestScore = score;
                    if (score > bestScoreEver) {
                        bestScoreEver = score;
                        currentBestAI = ai;
                    }
                }
            }
            if (watchLive) repaint();
        }
        evolveNextGeneration();
        generation++;

        if (hub != null) {
            hub.onGenerationComplete(3, genBestScore, bestScoreEver, currentBestAI.brain);
        }
    }

    private void executeAIMove(ChessAI ai) {
        double[] inputs = new double[BrainManager.INPUT_SIZE];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                inputs[r * 8 + c] = board[r][c] / 100.0;
            }
        }

        if (hub != null) {
            hub.getMoodController().injectMetaInputs(inputs, 3);
        }

        double[] outputs = ai.brain.predict(inputs);

        double maxVal = -Double.MAX_VALUE;
        for (int i = 0; i < 64; i++) {
            if (outputs[i] > maxVal) {
                maxVal = outputs[i];
            }
        }

        for (int r = 6; r >= 0; r--) {
            for (int c = 0; c < 8; c++) {
                if (board[r][c] > 0 && r > 0 && board[r - 1][c] <= 0) {
                    int piece = board[r][c];
                    board[r][c] = 0;
                    board[r - 1][c] = piece;
                    return;
                }
            }
        }
    }

    private int evaluateBoardMaterial() {
        int total = 0;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                if (board[r][c] > 0) total += board[r][c];
            }
        }
        return total;
    }

    private boolean isGameOver() {
        boolean whiteKing = false, blackKing = false;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                if (board[r][c] == 100) whiteKing = true;
                if (board[r][c] == -100) blackKing = true;
            }
        }
        return !whiteKing || !blackKing;
    }

    private void evolveNextGeneration() {
        population.sort((a, b) -> Double.compare(b.fitness, a.fitness));
        if (population.get(0).score >= currentBestAI.score) {
            currentBestAI = population.get(0);
        }
        ArrayList<ChessAI> nextGen = new ArrayList<>();
        nextGen.add(new ChessAI(population.get(0).brain));
        while (nextGen.size() < POPULATION_SIZE) {
            nextGen.add(population.get(0).cloneAndMutate());
        }
        population = nextGen;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int cellSize = 55;
        int startX = 30;
        int startY = 30;

        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                g.setColor((r + c) % 2 == 0 ? new Color(240, 217, 181) : new Color(181, 136, 99));
                g.fillRect(startX + c * cellSize, startY + r * cellSize, cellSize, cellSize);

                int piece = board[r][c];
                if (piece != 0) {
                    g.setColor(piece > 0 ? Color.WHITE : Color.BLACK);
                    g.fillOval(startX + c * cellSize + 8, startY + r * cellSize + 8, 39, 39);
                    g.setColor(piece > 0 ? Color.BLACK : Color.WHITE);
                    g.setFont(new Font("SansSerif", Font.BOLD, 13));
                    g.drawString(getPieceSymbol(piece), startX + c * cellSize + 20, startY + r * cellSize + 32);
                }
            }
        }

        g.setFont(new Font("SansSerif", Font.BOLD, 14));
        g.setColor(Color.WHITE);
        g.drawString("Mode: " + (playMode ? "Play vs AI" : "Auto-Training"), 490, 45);
        g.drawString("Press 'P' to Toggle Mode", 490, 70);
        g.drawString("Generation: " + generation, 490, 110);
        g.drawString("Best Material: " + bestScoreEver, 490, 135);
    }

    private String getPieceSymbol(int p) {
        int abs = Math.abs(p);
        if (abs == 1) return "P";
        if (abs == 3) return "N";
        if (abs == 5) return "R";
        if (abs == 9) return "Q";
        if (abs == 100) return "K";
        return "?";
    }
}