package tests.trace;

import ai.core.AI;
import ai.RandomBiasedAI;
import java.io.FileWriter;
import java.io.Writer;
import java.lang.reflect.Constructor;

import rts.*;
import rts.units.UnitTypeTable;
import util.XMLWriter;

/**
 * Records a game trace between two AIs (designed for LLM vs Reference AI games).
 *
 * Usage: java -cp "lib/*:bin" tests.trace.RecordLLMGame <AI1_class> <AI2_class> <output_prefix> [max_cycles]
 *
 * Example: java -cp "lib/*:bin" tests.trace.RecordLLMGame ai.abstraction.LLM_Gemini ai.RandomBiasedAI gemini_vs_random 3000
 */
public class RecordLLMGame {
    public static void main(String args[]) {
        // Parse arguments
        String ai1Class = args.length > 0 ? args[0] : "ai.abstraction.LLM_Gemini";
        String ai2Class = args.length > 1 ? args[1] : "ai.RandomBiasedAI";
        String outputPrefix = args.length > 2 ? args[2] : "game_trace";
        int maxCycles = args.length > 3 ? Integer.parseInt(args[3]) : 3000;
        String mapLocation = args.length > 4 ? args[4] : "maps/8x8/basesWorkers8x8.xml";

        System.out.println("=== Recording Game Trace ===");
        System.out.println("AI1: " + ai1Class);
        System.out.println("AI2: " + ai2Class);
        System.out.println("Map: " + mapLocation);
        System.out.println("Max Cycles: " + maxCycles);
        System.out.println("Output: " + outputPrefix + ".xml/.json");
        System.out.println();

        AI ai1 = null;
        AI ai2 = null;

        try {
            // Initialize game
            UnitTypeTable utt = new UnitTypeTable();
            PhysicalGameState pgs = PhysicalGameState.load(mapLocation, utt);
            GameState gs = new GameState(pgs, utt);

            // Create AIs
            try {
                Constructor<?> cons1 = Class.forName(ai1Class).getConstructor(UnitTypeTable.class);
                ai1 = (AI) cons1.newInstance(utt);
            } catch (Exception e) {
                System.out.println("ERROR: Failed to create AI1 (" + ai1Class + "): " + e.getMessage());
                e.printStackTrace(System.out);
                System.exit(1);
                return;
            }
            try {
                Constructor<?> cons2 = Class.forName(ai2Class).getConstructor(UnitTypeTable.class);
                ai2 = (AI) cons2.newInstance(utt);
            } catch (Exception e) {
                System.out.println("ERROR: Failed to create AI2 (" + ai2Class + "): " + e.getMessage());
                e.printStackTrace(System.out);
                System.exit(1);
                return;
            }

            System.out.println("AI1 instance: " + ai1);
            System.out.println("AI2 instance: " + ai2);
            System.out.println();

            // Initialize trace
            Trace trace = new Trace(utt);
            TraceEntry te = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
            trace.addEntry(te);

            // Game loop
            boolean gameover = false;
            int lastReportedTick = 0;
            int crashedPlayer = -1;

            System.out.println("Starting game...");

            do {
                // Report progress every 100 ticks
                if (gs.getTime() - lastReportedTick >= 100) {
                    System.out.println("  Tick: " + gs.getTime());
                    lastReportedTick = gs.getTime();
                }

                PlayerAction pa1 = null;
                PlayerAction pa2 = null;

                try {
                    pa1 = ai1.getAction(0, gs);
                } catch (Exception e) {
                    System.out.println("ERROR: AI1 crashed at tick " + gs.getTime() + ": " + e.getMessage());
                    e.printStackTrace(System.out);
                    crashedPlayer = 0;
                    break;
                }

                try {
                    pa2 = ai2.getAction(1, gs);
                } catch (Exception e) {
                    System.out.println("ERROR: AI2 crashed at tick " + gs.getTime() + ": " + e.getMessage());
                    e.printStackTrace(System.out);
                    crashedPlayer = 1;
                    break;
                }

                // Record non-empty actions
                if (!pa1.isEmpty() || !pa2.isEmpty()) {
                    te = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
                    te.addPlayerAction(pa1.clone());
                    te.addPlayerAction(pa2.clone());
                    trace.addEntry(te);
                }

                gs.issueSafe(pa1);
                gs.issueSafe(pa2);

                // Simulate
                gameover = gs.cycle();
            } while (!gameover && gs.getTime() < maxCycles);

            // Notify AIs of game over
            try { ai1.gameOver(gs.winner()); } catch (Exception e) { /* ignore */ }
            try { ai2.gameOver(gs.winner()); } catch (Exception e) { /* ignore */ }

            // Add final state to trace
            te = new TraceEntry(gs.getPhysicalGameState().clone(), gs.getTime());
            trace.addEntry(te);

            // Print result
            System.out.println();
            System.out.println("=== Game Result ===");
            System.out.println("Final tick: " + gs.getTime());

            if (crashedPlayer >= 0) {
                // Crashed AI loses, opponent wins
                int winner = (crashedPlayer == 0) ? 1 : 0;
                System.out.println("CRASHED: Player " + crashedPlayer);
                System.out.println("Winner: Player " + winner + " (opponent of crashed AI)");
            } else {
                int winner = gs.winner();
                if (winner == 0) {
                    System.out.println("Winner: Player 0 (" + ai1Class + ")");
                } else if (winner == 1) {
                    System.out.println("Winner: Player 1 (" + ai2Class + ")");
                } else {
                    System.out.println("Result: Draw");
                }
            }

            // Save trace as XML
            String xmlPath = outputPrefix + ".xml";
            XMLWriter xml = new XMLWriter(new FileWriter(xmlPath));
            trace.toxml(xml);
            xml.flush();
            xml.close();
            System.out.println("Saved XML trace: " + xmlPath);

            // Save trace as JSON
            String jsonPath = outputPrefix + ".json";
            Writer w = new FileWriter(jsonPath);
            trace.toJSON(w);
            w.flush();
            w.close();
            System.out.println("Saved JSON trace: " + jsonPath);

            System.out.println();
            System.out.println("To view the trace, run:");
            System.out.println("  java -cp \"lib/*:bin\" tests.trace.RecordLLMGame " + xmlPath);

        } catch (Exception e) {
            System.out.println("ERROR: Game failed: " + e.getMessage());
            e.printStackTrace(System.out);
            System.exit(1);
        }
    }
}
