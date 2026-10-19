package neural_network.master_network;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class MoodController {
    public static final int GAME_FLAPPY = 0;
    public static final int GAME_SNAKE = 1;
    public static final int GAME_OTHELLO = 2;
    public static final int GAME_CHESS = 3;

    public static final String[] GAME_NAMES = {"Flappy Bird", "Snake", "Othello", "Chess"};

    private static final int MIN_GENS_BEFORE_SWITCH = 10;

    private int currentGameId = GAME_FLAPPY;
    private int generationsInCurrentGame = 0;

    private double boredom = 0.05;
    private double frustration = 0.10;
    private double dopamine = 0.50;
    private double energy = 0.95;
    private double curiosity = 0.30;

    private double currentLoss = 0.0;
    private final List<Double> lossHistory = new ArrayList<>();

    private final int[] gamesPlayedCount = new int[4];
    private final int[] highScores = new int[4];
    private String lastDecisionReason = "Initializing Network...";

    private final BrainManager brainManager;
    private final Random random = new Random();

    public MoodController(BrainManager brainManager) {
        this.brainManager = brainManager;
    }

    public synchronized void injectMetaInputs(double[] inputs, int gameId) {
        inputs[64] = (gameId == GAME_FLAPPY) ? 1.0 : 0.0;
        inputs[65] = (gameId == GAME_SNAKE) ? 1.0 : 0.0;
        inputs[66] = (gameId == GAME_OTHELLO) ? 1.0 : 0.0;
        inputs[67] = (gameId == GAME_CHESS) ? 1.0 : 0.0;
        inputs[68] = boredom;
        inputs[69] = frustration;
        inputs[70] = dopamine;
        inputs[71] = energy;
        inputs[72] = curiosity;
        inputs[73] = Math.min(1.0, highScores[GAME_FLAPPY] / 50.0);
        inputs[74] = Math.min(1.0, highScores[GAME_SNAKE] / 30.0);
        inputs[75] = Math.min(1.0, highScores[GAME_OTHELLO] / 64.0);
        inputs[76] = Math.min(1.0, highScores[GAME_CHESS] / 100.0);
        inputs[77] = Math.min(1.0, generationsInCurrentGame / (double) MIN_GENS_BEFORE_SWITCH);
        inputs[78] = Math.min(1.0, gamesPlayedCount[GAME_FLAPPY] / 100.0);
        inputs[79] = Math.min(1.0, gamesPlayedCount[GAME_SNAKE] / 100.0);
        inputs[80] = Math.min(1.0, gamesPlayedCount[GAME_OTHELLO] / 100.0);
        inputs[81] = Math.min(1.0, gamesPlayedCount[GAME_CHESS] / 100.0);
        inputs[82] = 1.0;
    }

    public synchronized int evaluateGeneration(int gameId, int genScore, int maxScore, NeuralNetwork bestGenBrain) {
        brainManager.updateMasterNetwork(bestGenBrain);

        gamesPlayedCount[gameId]++;
        generationsInCurrentGame++;

        updateMoodStats(gameId, genScore, maxScore);

        NeuralNetwork masterNetwork = brainManager.getMasterNetwork();
        double[] decisionInputs = new double[BrainManager.INPUT_SIZE];
        injectMetaInputs(decisionInputs, gameId);
        double[] outputs = masterNetwork.predict(decisionInputs);

        double flappyDesire = outputs[64];
        double snakeDesire = outputs[65];
        double othelloDesire = outputs[66];
        double chessDesire = outputs[67];
        double switchImpulse = outputs[68];

        double[] scores = new double[4];
        scores[GAME_FLAPPY] = flappyDesire * 0.5 + energy * 0.3 + dopamine * 0.2;
        scores[GAME_SNAKE] = snakeDesire * 0.5 + curiosity * 0.3 + energy * 0.2;
        scores[GAME_OTHELLO] = othelloDesire * 0.5 + (1.0 - energy) * 0.35 + (1.0 - frustration) * 0.15;
        scores[GAME_CHESS] = chessDesire * 0.5 + (1.0 - energy) * 0.25 + curiosity * 0.25;

        scores[currentGameId] -= (boredom * 0.35 + frustration * 0.30);

        int chosenGame = 0;
        for (int i = 1; i < 4; i++) {
            if (scores[i] > scores[chosenGame]) chosenGame = i;
        }

        boolean wantsToSwitch = (generationsInCurrentGame >= MIN_GENS_BEFORE_SWITCH) &&
                (chosenGame != currentGameId) && (switchImpulse > 0.72 || boredom > 0.75 || frustration > 0.80);

        double[] targetOutputs = outputs.clone();
        if (wantsToSwitch) {
            targetOutputs[64 + chosenGame] = 0.90;
            targetOutputs[64 + currentGameId] = 0.20;
            targetOutputs[68] = 0.85;
        } else {
            targetOutputs[64 + currentGameId] = clamp(0.65 + dopamine * 0.30 - frustration * 0.25);
            targetOutputs[68] = clamp(boredom * 0.5 + frustration * 0.5);
        }

        currentLoss = masterNetwork.trainSingleEpoch(new double[][]{decisionInputs}, new double[][]{targetOutputs});
        lossHistory.add(currentLoss);
        if (lossHistory.size() > 60) {
            lossHistory.remove(0);
        }

        if (wantsToSwitch) {
            lastDecisionReason = String.format("Switched to %s after %d gens", GAME_NAMES[chosenGame], generationsInCurrentGame);

            boredom = clamp(boredom - 0.65);
            frustration = clamp(frustration - 0.50);
            curiosity = clamp(curiosity - 0.30);
            generationsInCurrentGame = 0;
            currentGameId = chosenGame;

            return chosenGame;
        } else {
            lastDecisionReason = String.format("Staying on %s (Gen %d/%d)", GAME_NAMES[currentGameId], generationsInCurrentGame, MIN_GENS_BEFORE_SWITCH);
            return -1;
        }
    }

    private void updateMoodStats(int gameId, int genScore, int maxScore) {
        boolean beatRecord = maxScore > highScores[gameId];
        if (beatRecord) highScores[gameId] = maxScore;

        boredom = clamp(boredom + 0.03 + random.nextDouble() * 0.02);
        curiosity = clamp(curiosity + 0.025 + (random.nextDouble() - 0.45) * 0.03);

        boolean goodPerformance;
        if (gameId == GAME_SNAKE) {
            goodPerformance = beatRecord || genScore > 0 || (maxScore > 0 && genScore >= maxScore * 0.4);
        } else if (gameId == GAME_CHESS) {
            goodPerformance = beatRecord || (maxScore > 0 && genScore >= maxScore * 0.5);
        } else {
            goodPerformance = beatRecord || (maxScore > 0 && genScore >= maxScore * 0.7);
        }

        if (goodPerformance) {
            dopamine = clamp(dopamine + 0.12);
            frustration = clamp(frustration - 0.10);
        } else {
            dopamine = clamp(dopamine - 0.03);
            double frustrationGain = (gameId == GAME_SNAKE || gameId == GAME_CHESS) ? 0.025 : 0.045;
            frustration = clamp(frustration + frustrationGain);
        }

        if (gameId == GAME_FLAPPY || gameId == GAME_SNAKE) {
            energy = clamp(energy - 0.035);
        } else {
            energy = clamp(energy + 0.08);
        }
    }

    private double clamp(double val) {
        return Math.max(0.0, Math.min(1.0, val));
    }

    public int getCurrentGameId() { return currentGameId; }
    public void setCurrentGameId(int currentGameId) { this.currentGameId = currentGameId; }
    public String getLastDecisionReason() { return lastDecisionReason; }
    public void setLastDecisionReason(String lastDecisionReason) { this.lastDecisionReason = lastDecisionReason; }
    public double getBoredom() { return boredom; }
    public double getFrustration() { return frustration; }
    public double getDopamine() { return dopamine; }
    public double getEnergy() { return energy; }
    public double getCuriosity() { return curiosity; }
    public double getCurrentLoss() { return currentLoss; }
    public synchronized List<Double> getLossHistory() { return new ArrayList<>(lossHistory); }
    public int[] getHighScores() { return highScores; }
    public int[] getGamesPlayedCount() { return gamesPlayedCount; }
}