package neural_network.master_network;

import javax.swing.*;
import java.awt.*;
import java.util.List;

public class TelemetryPanel extends JPanel {
    private final MoodController moodController;

    public TelemetryPanel(MoodController moodController) {
        this.moodController = moodController;
        this.setPreferredSize(new Dimension(275, 790));
        this.setBackground(new Color(22, 24, 34));
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D) g;
        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        int activeGame = moodController.getCurrentGameId();
        g2.setColor(Color.WHITE);
        g2.setFont(new Font("SansSerif", Font.BOLD, 15));
        g2.drawString("STATS & TELEMETRY", 18, 28);

        g2.setFont(new Font("SansSerif", Font.BOLD, 13));
        g2.setColor(new Color(0, 220, 255));
        g2.drawString("Playing: " + MoodController.GAME_NAMES[activeGame], 18, 52);

        g2.setFont(new Font("SansSerif", Font.PLAIN, 11));
        g2.setColor(new Color(180, 220, 255));
        g2.drawString(moodController.getLastDecisionReason(), 18, 72);

        int startY = 105;
        int gapY = 48;
        drawStatBar(g2, "Boredom", moodController.getBoredom(), startY, new Color(245, 158, 11));
        drawStatBar(g2, "Frustration", moodController.getFrustration(), startY + gapY, new Color(239, 68, 68));
        drawStatBar(g2, "Dopamine", moodController.getDopamine(), startY + gapY * 2, new Color(16, 185, 129));
        drawStatBar(g2, "Energy", moodController.getEnergy(), startY + gapY * 3, new Color(59, 130, 246));
        drawStatBar(g2, "Curiosity", moodController.getCuriosity(), startY + gapY * 4, new Color(168, 85, 247));

        int recY = 365;
        g2.setColor(Color.WHITE);
        g2.setFont(new Font("SansSerif", Font.BOLD, 13));
        g2.drawString("HIGH SCORES & SESSIONS", 18, recY);

        int[] highScores = moodController.getHighScores();
        int[] counts = moodController.getGamesPlayedCount();
        g2.setFont(new Font("SansSerif", Font.PLAIN, 12));
        g2.setColor(Color.LIGHT_GRAY);
        for (int i = 0; i < MoodController.GAME_NAMES.length; i++) {
            g2.drawString(String.format("• %-11s Max: %-3d (Gens: %d)",
                    MoodController.GAME_NAMES[i], highScores[i], counts[i]), 18, recY + 24 + i * 22);
        }

        int lossY = 470;
        g2.setColor(Color.WHITE);
        g2.setFont(new Font("SansSerif", Font.BOLD, 13));
        g2.drawString("LOSS FUNCTION (MSE)", 18, lossY);

        g2.setFont(new Font("Monospaced", Font.BOLD, 12));
        g2.setColor(new Color(255, 120, 120));
        g2.drawString(String.format("Current Loss: %.6f", moodController.getCurrentLoss()), 18, lossY + 22);

        drawLossGraph(g2, lossY + 35);
    }

    private void drawStatBar(Graphics2D g2, String label, double value, int y, Color color) {
        g2.setColor(Color.WHITE);
        g2.setFont(new Font("SansSerif", Font.PLAIN, 12));
        g2.drawString(label + ": " + (int) (value * 100) + "%", 18, y);

        g2.setColor(new Color(45, 52, 68));
        g2.fillRoundRect(18, y + 6, 235, 14, 8, 8);

        g2.setColor(color);
        g2.fillRoundRect(18, y + 6, (int) (235 * value), 14, 8, 8);
    }

    private void drawLossGraph(Graphics2D g2, int y) {
        g2.setColor(new Color(14, 16, 24));
        g2.fillRoundRect(18, y, 235, 140, 8, 8);
        g2.setColor(new Color(55, 65, 85));
        g2.drawRoundRect(18, y, 235, 140, 8, 8);

        List<Double> history = moodController.getLossHistory();
        if (history.size() < 2) {
            g2.setColor(Color.GRAY);
            g2.setFont(new Font("SansSerif", Font.ITALIC, 11));
            g2.drawString("Waiting for generation data...", 18 + 35, y + 140 / 2);
            return;
        }

        double maxLoss = 0.0001;
        for (double val : history) {
            if (val > maxLoss) maxLoss = val;
        }

        g2.setColor(new Color(239, 68, 68));
        g2.setStroke(new BasicStroke(2f));

        int n = history.size();
        for (int i = 0; i < n - 1; i++) {
            int x1 = 18 + 6 + (i * (235 - 12)) / (n - 1);
            int x2 = 18 + 6 + ((i + 1) * (235 - 12)) / (n - 1);

            int y1 = y + 140 - 8 - (int) ((history.get(i) / maxLoss) * (140 - 16));
            int y2 = y + 140 - 8 - (int) ((history.get(i + 1) / maxLoss) * (140 - 16));

            g2.drawLine(x1, y1, x2, y2);
        }
    }
}