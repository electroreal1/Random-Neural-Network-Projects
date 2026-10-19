package neural_network;

import neural_network.games.ChessGamePanel;
import neural_network.games.FlappyBirdGame;
import neural_network.games.OthelloGamePanel;
import neural_network.games.SnakeGame;
import neural_network.master_network.*;

import javax.swing.*;
import java.awt.*;

public class Main extends JFrame {
    private final BrainManager brainManager;
    private final MoodController moodController;
    private final TelemetryPanel telemetryPanel;
    private final NetworkVisualizerPanel visualizerPanel;

    private final CardLayout cardLayout = new CardLayout();
    private final JPanel gameContainer = new JPanel(cardLayout);

    private final FlappyBirdGame flappyGame;
    private final SnakeGame snakeGame;
    private final OthelloGamePanel othelloGame;
    private final ChessGamePanel chessGame;

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            Main main = new Main();
            main.setVisible(true);
        });
    }

    public Main() {
        this.setTitle("Neural Network - Autonomous Multi-Game Dashboard");
        this.setSize(1300, 780);
        this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        this.setLocationRelativeTo(null);
        this.setLayout(new BorderLayout());

        this.brainManager = new BrainManager();
        this.moodController = new MoodController(brainManager);

        this.visualizerPanel = new NetworkVisualizerPanel(brainManager);
        this.telemetryPanel = new TelemetryPanel(moodController);

        this.add(visualizerPanel, BorderLayout.WEST);
        this.add(gameContainer, BorderLayout.CENTER);
        this.add(telemetryPanel, BorderLayout.EAST);

        this.flappyGame = new FlappyBirdGame(this);
        this.snakeGame = new SnakeGame(this);
        this.othelloGame = new OthelloGamePanel(this);
        this.chessGame = new ChessGamePanel(this);

        this.gameContainer.add(flappyGame, MoodController.GAME_NAMES[MoodController.GAME_FLAPPY]);
        this.gameContainer.add(snakeGame, MoodController.GAME_NAMES[MoodController.GAME_SNAKE]);
        this.gameContainer.add(othelloGame, MoodController.GAME_NAMES[MoodController.GAME_OTHELLO]);
        this.gameContainer.add(chessGame, MoodController.GAME_NAMES[MoodController.GAME_CHESS]);

        this.flappyGame.startGame();
        this.snakeGame.startGame();
        this.othelloGame.startGame();
        this.chessGame.startGame();

        new Timer(40, e -> visualizerPanel.repaint()).start();

        this.switchToGame(MoodController.GAME_FLAPPY, "Starting initial session in Flappy Bird");
    }

    public void onGenerationComplete(int gameId, int genScore, int maxScore, NeuralNetwork bestGenBrain) {
        int nextGame = moodController.evaluateGeneration(gameId, genScore, maxScore, bestGenBrain);
        if (nextGame != -1) {
            SwingUtilities.invokeLater(() -> this.switchToGame(nextGame, moodController.getLastDecisionReason()));
        } else {
            telemetryPanel.repaint();
        }
    }

    private void switchToGame(int gameId, String reason) {
        moodController.setCurrentGameId(gameId);
        moodController.setLastDecisionReason(reason);

        flappyGame.setActive(gameId == MoodController.GAME_FLAPPY);
        snakeGame.setActive(gameId == MoodController.GAME_SNAKE);
        othelloGame.setActive(gameId == MoodController.GAME_OTHELLO);
        chessGame.setActive(gameId == MoodController.GAME_CHESS);

        cardLayout.show(gameContainer, MoodController.GAME_NAMES[gameId]);
        telemetryPanel.repaint();
        visualizerPanel.repaint();
    }

    public BrainManager getBrainManager() {
        return brainManager;
    }

    public MoodController getMoodController() {
        return moodController;
    }
}