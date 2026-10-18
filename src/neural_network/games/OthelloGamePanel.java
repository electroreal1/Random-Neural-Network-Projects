package neural_network.games;

import neural_network.Layer;
import neural_network.NeuralNetwork;
import neural_network.Neuron;

import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class OthelloGamePanel extends JPanel implements Runnable {
    private Thread gameThread;
    private boolean isRunning = true;

    private final int POPULATION_SIZE = 30;
    private ArrayList<OthelloAI> population;
    private int generation = 1;
    private int bestScoreEver = 0;

    private int[][] board = new int[8][8];
    private boolean aiTurn = true; // True = AI (Black / 1), False = Human (White / 2)
    private boolean playMode = false;
    private Random random = new Random();
    private OthelloAI currentBestAI;

    private final String weightsFile = "othello_weights.txt";

    private class OthelloAI {
        NeuralNetwork brain;
        double fitness = 0;
        int wins = 0;

        OthelloAI(NeuralNetwork brain) {
            this.brain = brain;
        }

        OthelloAI cloneAndMutate() {
            NeuralNetwork mutantBrain = new NeuralNetwork(0.1);
            mutantBrain.addLayer(new Layer(32, 64));
            mutantBrain.addLayer(new Layer(64, 32));

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
            return new OthelloAI(mutantBrain);
        }
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Neural Othello Evolution Arena (Press 'P' to Play vs AI)");
        OthelloGamePanel game = new OthelloGamePanel();
        frame.add(game);
        frame.setSize(600, 650);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
        game.startGame();
    }

    public OthelloGamePanel() {
        setBackground(new Color(20, 80, 40));
        setFocusable(true);

        initPopulation();

        File f = new File(weightsFile);
        if (f.exists() && !population.isEmpty()) {
            population.get(0).brain.loadWeights(weightsFile);
            currentBestAI = population.get(0);
        }

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            if (currentBestAI != null) {
                currentBestAI.brain.saveWeights(weightsFile);
                System.out.println("Auto-saved best Othello AI weights to " + weightsFile);
            }
        }));

        addMouseListener(new MouseAdapter() {
            @Override
            public void mousePressed(MouseEvent e) {
                if (playMode && !aiTurn && !isGameOver()) {
                    int cellSize = 60;
                    int startX = 30;
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

        addKeyListener(new java.awt.event.KeyAdapter() {
            @Override
            public void keyPressed(java.awt.event.KeyEvent e) {
                if (e.getKeyCode() == java.awt.event.KeyEvent.VK_P) {
                    playMode = !playMode;
                    resetBoard();
                    System.out.println("Mode changed: " + (playMode ? "Play vs AI Mode" : "Auto-Training Mode"));
                }
            }
        });

        resetBoard();
    }

    private void initPopulation() {
        population = new ArrayList<>();
        for (int i = 0; i < POPULATION_SIZE; i++) {
            NeuralNetwork net = new NeuralNetwork(0.1);
            net.addLayer(new Layer(32, 64));
            net.addLayer(new Layer(64, 32));
            initializeRandomWeights(net);
            population.add(new OthelloAI(net));
        }
        currentBestAI = population.get(0);
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
        while (isRunning) {
            if (!playMode) {
                playGeneration();
                try {
                    Thread.sleep(30);
                } catch (InterruptedException e) {
                    break;
                }
            } else {
                if (aiTurn && !isGameOver()) {
                    if (hasValidMoves(1)) {
                        int move = getBestValidMove(currentBestAI, 1);
                        if (move != -1) {
                            makeMove(move / 8, move % 8, 1);
                        }
                    }

                    if (hasValidMoves(2)) {
                        aiTurn = false;
                    } else if (hasValidMoves(1)) {
                        aiTurn = true;
                    }
                    repaint();
                }
                try {
                    Thread.sleep(600);
                } catch (InterruptedException e) {
                    break;
                }
            }
        }
    }

    private void playGeneration() {
        for (OthelloAI ai : population) {
            resetBoard();
            boolean gameOver = false;

            while (!gameOver) {
                if (aiTurn) {
                    if (hasValidMoves(1)) {
                        int move = getBestValidMove(ai, 1);
                        if (move != -1) {
                            makeMove(move / 8, move % 8, 1);
                            ai.fitness += 5;
                        }
                    } else {
                        ai.fitness -= 2;
                    }

                    if (hasValidMoves(2)) {
                        aiTurn = false;
                    } else if (hasValidMoves(1)) {
                        aiTurn = true;
                    }
                } else {
                    if (hasValidMoves(2)) {
                        List<Integer> validMoves = getValidMoves(2);
                        int move = validMoves.get(random.nextInt(validMoves.size()));
                        makeMove(move / 8, move % 8, 2);
                    }

                    if (hasValidMoves(1)) {
                        aiTurn = true;
                    } else if (hasValidMoves(2)) {
                        aiTurn = false;
                    }
                }

                if (isGameOver()) {
                    gameOver = true;
                    int aiScore = countPieces(1);
                    ai.fitness += aiScore * 10;
                    ai.wins = aiScore;
                    if (aiScore > bestScoreEver) {
                        bestScoreEver = aiScore;
                        currentBestAI = ai;
                    }
                }
            }
        }
        evolveNextGeneration();
        generation++;
    }

    private int getBestValidMove(OthelloAI ai, int player) {
        double[] inputs = new double[64];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 8; c++) {
                inputs[r * 8 + c] = board[r][c] == player ? 1.0 : (board[r][c] == 0 ? 0.0 : -1.0);
            }
        }

        double[] outputs = ai.brain.predict(inputs);
        List<Integer> validMoves = getValidMoves(player);

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
                for (Point p : toFlip) {
                    board[p.x][p.y] = player;
                }
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
        int startX = 30;
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

        // Highlight valid possible moves in Play Mode
        if (playMode && !isGameOver()) {
            int activePlayer = aiTurn ? 1 : 2;
            List<Integer> validMoves = getValidMoves(activePlayer);
            g.setColor(new Color(0, 255, 100, 140)); // Translucent green dot
            for (int move : validMoves) {
                int r = move / 8;
                int c = move % 8;
                g.fillOval(startX + c * cellSize + 22, startY + r * cellSize + 22, 16, 16);
            }
        }

        g.setFont(new Font("SansSerif", Font.BOLD, 15));
        g.setColor(Color.WHITE);
        g.drawString("Mode: " + (playMode ? "🎮 Play vs AI" : "🧬 Auto-Training"), 510, 30);
        g.drawString("Press 'P' to Toggle Mode", 510, 55);
        g.drawString("Generation: " + generation, 510, 95);
        g.drawString("Best Score Ever: " + bestScoreEver, 510, 125);
        g.drawString("AI (Black): " + countPieces(1), 510, 175);
        g.drawString("Human (White): " + countPieces(2), 510, 205);

        if (playMode) {
            g.setColor(aiTurn ? Color.YELLOW : Color.CYAN);
            g.drawString("Turn: " + (aiTurn ? "AI (Thinking...)" : "Your Turn"), 510, 240);
        }

        if (isGameOver()) {
            g.setFont(new Font("SansSerif", Font.BOLD, 20));
            g.setColor(Color.YELLOW);
            g.drawString("GAME OVER!", 510, 280);
            int black = countPieces(1);
            int white = countPieces(2);
            String winner = black > white ? "AI Wins!" : (white > black ? "You Win!" : "Tie Game!");
            g.drawString(winner, 510, 310);
        }
    }
}