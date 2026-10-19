package neural_network.games;

import neural_network.master_network.BrainManager;
import neural_network.Main;
import neural_network.master_network.MoodController;
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

public class OthelloGamePanel extends JPanel implements Runnable {
    private volatile boolean active = true;
    private final Main hub;
    private final int POPULATION_SIZE = 2;
    private static final int AUTO_MOVE_DELAY_MS = 200;
    private static final int HUMAN_MODE_DELAY_MS = 650;

    private ArrayList<OthelloAI> population;
    private int generation = 1;
    private int bestScoreEver = 0;

    private final int[][] board = new int[8][8];
    private boolean aiTurn = true;
    private boolean playMode = false;
    private final Random random = new Random();
    private OthelloAI currentBestAI;

    private static class OthelloAI {
        NeuralNetwork brain;
        double fitness = 0;
        int wins = 0;

        OthelloAI(NeuralNetwork brain) {
            this.brain = brain;
        }

        OthelloAI cloneAndMutate() {
            return new OthelloAI(BrainManager.cloneAndMutate(this.brain));
        }
    }

    public OthelloGamePanel(Main hub) {
        this.hub = hub;
        setBackground(new Color(20, 80, 40));
        setFocusable(true);

        initPopulation();

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                requestFocusInWindow();
                if (playMode && !aiTurn && !isGameOver()) {
                    int cellSize = 60;
                    int startX = 20;
                    int startY = 30;

                    int c = (e.getX() - startX) / cellSize;
                    int r = (e.getY() - startY) / cellSize;

                    if (r >= 0 && r < 8 && c >= 0 && c < 8) {
                        if (isValidMove(r, c, 2)) {
                            makeMove(r, c, 2);

                            if (hasValidMoves(1)) {
                                aiTurn = true;
                            } else if (hasValidMoves(2)) {
                                aiTurn = false;
                            }
                            repaint();
                        }
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
            population.add(new OthelloAI(net));
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
        board[3][3] = 2;
        board[3][4] = 1;
        board[4][3] = 1;
        board[4][4] = 2;
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
                    if (hasValidMoves(1)) {
                        int move = getBestValidMove(currentBestAI);
                        if (move != -1) {
                            makeMove(move / 8, move % 8, 1);
                        }
                    }

                    if (hasValidMoves(2)) {
                        aiTurn = false;
                    } else if (hasValidMoves(1)) aiTurn = true;
                    repaint();
                }
                try { Thread.sleep(HUMAN_MODE_DELAY_MS); } catch (InterruptedException e) { break; }
            }
        }
    }

    private void playGeneration() {
        int genBestScore = 0;
        for (int i = 0; i < population.size(); i++) {
            if (!active || playMode) return;

            OthelloAI ai = population.get(i);
            boolean watchLive = (i == 0);
            resetBoard();
            boolean gameOver = false;

            while (!gameOver && active && !playMode) {
                if (aiTurn) {
                    if (hasValidMoves(1)) {
                        int move = getBestValidMove(ai);
                        if (move != -1) {
                            makeMove(move / 8, move % 8, 1);
                            ai.fitness += 5;
                        }
                    } else ai.fitness -= 2;


                    if (hasValidMoves(2)) {
                        aiTurn = false;
                    } else if (hasValidMoves(1)) aiTurn = true;
                } else {
                    if (hasValidMoves(2)) {
                        List<Integer> validMoves = getValidMoves(2);
                        int move = validMoves.get(random.nextInt(validMoves.size()));
                        makeMove(move / 8, move % 8, 2);
                    }

                    if (hasValidMoves(1)) {
                        aiTurn = true;
                    } else if (hasValidMoves(2)) aiTurn = false;
                }

                if (watchLive) {
                    repaint();
                    try {
                        Thread.sleep(AUTO_MOVE_DELAY_MS);
                    } catch (InterruptedException e) {
                        return;
                    }
                }

                if (isGameOver()) {
                    gameOver = true;
                    int aiScore = countPieces(1);
                    ai.fitness += aiScore * 10;
                    ai.wins = aiScore;
                    if (aiScore > genBestScore) genBestScore = aiScore;
                    if (aiScore > bestScoreEver) {
                        bestScoreEver = aiScore;
                        currentBestAI = ai;
                    }
                }
            }
        }
        evolveNextGeneration();
        generation++;

        if (hub != null) {
            hub.onGenerationComplete(MoodController.GAME_OTHELLO, genBestScore, bestScoreEver, currentBestAI.brain);
        }
    }

    private int getBestValidMove(OthelloAI ai) {
        double[] inputs = new double[BrainManager.INPUT_SIZE];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                inputs[r * 8 + c] = board[r][c] == 1 ? 1.0 : (board[r][c] == 0 ? 0.0 : -1.0);
            }
        }

        if (hub != null) {
            hub.getMoodController().injectMetaInputs(inputs, MoodController.GAME_OTHELLO);
        }

        double[] outputs = ai.brain.predict(inputs);
        if (hub != null) {
            hub.getBrainManager().recordLiveState(ai.brain, inputs);
        }

        List<Integer> validMoves = getValidMoves(1);

        if (validMoves.isEmpty()) return -1;

        int bestMove = validMoves.get(0);
        double maxVal = -Double.MAX_VALUE;

        for (int move : validMoves) {
            if (outputs[move] > maxVal) {
                maxVal = outputs[move];
                bestMove = move;
            }
        }
        return bestMove;
    }

    private boolean hasValidMoves(int player) {
        return !getValidMoves(player).isEmpty();
    }

    private List<Integer> getValidMoves(int player) {
        List<Integer> moves = new ArrayList<>();
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                if (isValidMove(r, c, player)) {
                    moves.add(r * 8 + c);
                }
            }
        }
        return moves;
    }

    private boolean isValidMove(int r, int c, int player) {
        if (board[r][c] != 0) return false;
        int opponent = (player == 1) ? 2 : 1;
        int[] dr = {-1, -1, -1, 0, 0, 1, 1, 1};
        int[] dc = {-1, 0, 1, -1, 1, -1, 0, 1};

        for (int i = 0; i < 8; i++) {
            int nr = r + dr[i], nc = c + dc[i];
            boolean hasOpponentBetween = false;
            while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8 && board[nr][nc] == opponent) {
                nr += dr[i];
                nc += dc[i];
                hasOpponentBetween = true;
            }
            if (hasOpponentBetween && nr >= 0 && nr < 8 && nc >= 0 && nc < 8 && board[nr][nc] == player) {
                return true;
            }
        }
        return false;
    }

    private void makeMove(int r, int c, int player) {
        board[r][c] = player;
        int opponent = (player == 1) ? 2 : 1;
        int[] dr = {-1, -1, -1, 0, 0, 1, 1, 1};
        int[] dc = {-1, 0, 1, -1, 1, -1, 0, 1};

        for (int i = 0; i < 8; i++) {
            int nr = r + dr[i], nc = c + dc[i];
            List<Point> toFlip = new ArrayList<>();
            while (nr >= 0 && nr < 8 && nc >= 0 && nc < 8 && board[nr][nc] == opponent) {
                toFlip.add(new Point(nr, nc));
                nr += dr[i];
                nc += dc[i];
            }
            if (!toFlip.isEmpty() && nr >= 0 && nr < 8 && nc >= 0 && nc < 8 && board[nr][nc] == player) {
                for (Point p : toFlip) board[p.x][p.y] = player;
            }
        }
    }

    private boolean isGameOver() {
        return !hasValidMoves(1) && !hasValidMoves(2);
    }

    private int countPieces(int player) {
        int count = 0;
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                if (board[r][c] == player) count++;
            }
        }
        return count;
    }

    private void evolveNextGeneration() {
        population.sort((a, b) -> Double.compare(b.fitness, a.fitness));
        if (population.get(0).wins >= currentBestAI.wins) currentBestAI = population.get(0);
        ArrayList<OthelloAI> nextGen = new ArrayList<>();
        nextGen.add(new OthelloAI(population.get(0).brain));

        while (nextGen.size() < POPULATION_SIZE) {
            nextGen.add(population.get(0).cloneAndMutate());
        }
        population = nextGen;
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int cellSize = 60;
        int startX = 20;
        int startY = 30;

        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                g.setColor(new Color(30, 100, 50));
                g.fillRect(startX + c * cellSize, startY + r * cellSize, cellSize, cellSize);
                g.setColor(Color.BLACK);
                g.drawRect(startX + c * cellSize, startY + r * cellSize, cellSize, cellSize);

                if (board[r][c] == 1) {
                    g.setColor(Color.BLACK);
                    g.fillOval(startX + c * cellSize + 8, startY + r * cellSize + 8, 44, 44);
                } else if (board[r][c] == 2) {
                    g.setColor(Color.WHITE);
                    g.fillOval(startX + c * cellSize + 8, startY + r * cellSize + 8, 44, 44);
                }
            }
        }

        if (!isGameOver()) {
            int activePlayer = aiTurn ? 1 : 2;
            List<Integer> validMoves = getValidMoves(activePlayer);
            g.setColor(new Color(0, 255, 100, 140));
            for (int move : validMoves) {
                int r = move / 8;
                int c = move % 8;
                g.fillOval(startX + c * cellSize + 22, startY + r * cellSize + 22, 16, 16);
            }
        }

        g.setFont(new Font("SansSerif", Font.BOLD, 14));
        g.setColor(Color.WHITE);
        g.drawString("Mode: " + (playMode ? "Play vs AI" : "Auto-Training"), 510, 45);
        g.drawString("Click board + 'P' to toggle", 510, 70);
        g.drawString("Generation: " + generation, 510, 110);
        g.drawString("Best Score: " + bestScoreEver, 510, 135);
        g.drawString("AI (Black): " + countPieces(1), 510, 175);
        g.drawString("White: " + countPieces(2), 510, 200);
    }
}