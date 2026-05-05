package ai.abstraction.submissions.ai4pc;

import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.HeavyRush;
import ai.abstraction.LightRush;
import ai.abstraction.RangedRush;
import ai.abstraction.WorkerRushPlusPlus;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;
import java.util.*;
import com.google.gson.*;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.*;

public class AI4PC extends AbstractionLayerAI {

    static final String OLLAMA_HOST = System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    static final String MODEL = System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
    static final int LLM_INTERVAL = 100;

    private static final String INSTRUCTION = "You are choosing a combat doctrine for a MicroRTS bot on an 8x8 map.\n\n"
            +
            "Pick exactly ONE label and reply with that label only. No explanation.\n\n" +
            "Decision procedure (apply in order, take the FIRST that matches):\n\n" +
            "1. If enemy_heavy >= 1: pick RANGED_RUSH. Heavy units are slow; ranged kites them.\n\n" +
            "2. If enemy_light >= 2: pick HEAVY_RUSH. Heavy outranges and out-tanks Light.\n\n" +
            "3. If enemy_ranged >= 1: pick LIGHT_RUSH. Light is fast enough to close on ranged units before they kite.\n\n"
            +
            "4. If tick < 200 AND enemy has NO military units (enemy_light, enemy_heavy, enemy_ranged are all 0): pick WORKER_RUSH. Reason: on 8x8, the enemy's workers reach our base before any tech is ready. Mirroring with workers is the only safe opening.\n\n"
            +
            "5. Otherwise (mid-game, enemy still has only workers): pick LIGHT_RUSH.\n\n" +
            "STICKINESS RULE (apply AFTER the above):\n" +
            "After computing your answer, compare it to `previous_strategy` from the state. If they match, output that. If they differ, ONLY switch if the enemy has a unit type now that they didn't before — otherwise output `previous_strategy`. Switching mid-game without a real reason throws away units we've already built.\n\n"
            +
            "Labels:\n  WORKER_RUSH, LIGHT_RUSH, HEAVY_RUSH, RANGED_RUSH\n\n" +
            "Output exactly one label.\n\n" +
            "STATE\n";

    enum Strategy {
        WORKER_RUSH, LIGHT_RUSH, HEAVY_RUSH, RANGED_RUSH
    }

    protected UnitTypeTable utt;
    UnitType workerType, lightType, heavyType, rangedType, baseType, barracksType, resourceType;

    private final WorkerRushPlusPlus workerRush;
    private final LightRush lightRush;
    private final HeavyRush heavyRush;
    private final RangedRush rangedRush;

    private Strategy current = Strategy.WORKER_RUSH;
    private String lastEnemyComp = "";

    public AI4PC(UnitTypeTable a_utt) {
        this(a_utt, new AStarPathFinding());
    }

    public AI4PC(UnitTypeTable a_utt, PathFinding pf) {
        super(pf);
        reset(a_utt);
        this.workerRush = new WorkerRushPlusPlus(a_utt);
        this.lightRush = new LightRush(a_utt);
        this.heavyRush = new HeavyRush(a_utt);
        this.rangedRush = new RangedRush(a_utt);
    }

    public void reset() {
        super.reset();
        if (workerRush != null)
            workerRush.reset();
        if (lightRush != null)
            lightRush.reset();
        if (heavyRush != null)
            heavyRush.reset();
        if (rangedRush != null)
            rangedRush.reset();
        current = Strategy.WORKER_RUSH;
        lastEnemyComp = "";
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        workerType = utt.getUnitType("Worker");
        lightType = utt.getUnitType("Light");
        heavyType = utt.getUnitType("Heavy");
        rangedType = utt.getUnitType("Ranged");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
        resourceType = utt.getUnitType("Resource");
    }

    public AI clone() {
        return new AI4PC(utt, pf);
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if (gs.getTime() % LLM_INTERVAL == 0) {
            Strategy proposed = pickStrategy(player, gs);
            String enemyComp = computeEnemyComp(player, gs);
            boolean enemyChanged = !enemyComp.equals(lastEnemyComp);
            if (proposed == current || enemyChanged) {
                current = proposed;
            }
            lastEnemyComp = enemyComp;
            System.out.println("[AI4PC] tick=" + gs.getTime()
                    + " strategy=" + current + " (proposed=" + proposed
                    + ", enemyComp=" + enemyComp + ")");
        }
        return delegate(current).getAction(player, gs);
    }

    private String computeEnemyComp(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int eL = 0, eH = 0, eR = 0, eB = 0;
        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() < 0 || u.getPlayer() == player)
                continue;
            if (u.getType() == lightType)
                eL = 1;
            else if (u.getType() == heavyType)
                eH = 1;
            else if (u.getType() == rangedType)
                eR = 1;
            else if (u.getType() == barracksType)
                eB = 1;
        }
        return "L=" + eL + ",H=" + eH + ",R=" + eR + ",B=" + eB;
    }

    private AI delegate(Strategy s) {
        switch (s) {
            case LIGHT_RUSH:
                return lightRush;
            case HEAVY_RUSH:
                return heavyRush;
            case RANGED_RUSH:
                return rangedRush;
            default:
                return workerRush;
        }
    }

    private Strategy pickStrategy(int player, GameState gs) {
        String state = buildStateBlock(player, gs);
        String response = callOllama(state);
        return parseStrategy(response);
    }

    private String buildStateBlock(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);

        int myWorkers = 0, myLight = 0, myHeavy = 0, myRanged = 0, myBarracks = 0;
        int enWorkers = 0, enLight = 0, enHeavy = 0, enRanged = 0, enBarracks = 0;
        int myBaseX = -1, myBaseY = -1;

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                if (u.getType() == workerType)
                    myWorkers++;
                else if (u.getType() == lightType)
                    myLight++;
                else if (u.getType() == heavyType)
                    myHeavy++;
                else if (u.getType() == rangedType)
                    myRanged++;
                else if (u.getType() == barracksType)
                    myBarracks++;
                else if (u.getType() == baseType) {
                    myBaseX = u.getX();
                    myBaseY = u.getY();
                }
            } else if (u.getPlayer() >= 0) {
                if (u.getType() == workerType)
                    enWorkers++;
                else if (u.getType() == lightType)
                    enLight++;
                else if (u.getType() == heavyType)
                    enHeavy++;
                else if (u.getType() == rangedType)
                    enRanged++;
                else if (u.getType() == barracksType)
                    enBarracks++;
            }
        }

        int closestEnemy = -1;
        if (myBaseX >= 0) {
            int best = Integer.MAX_VALUE;
            for (Unit u : pgs.getUnits()) {
                if (u.getPlayer() < 0 || u.getPlayer() == player)
                    continue;
                if (!u.getType().canAttack)
                    continue;
                int d = Math.abs(u.getX() - myBaseX) + Math.abs(u.getY() - myBaseY);
                if (d < best)
                    best = d;
            }
            if (best != Integer.MAX_VALUE)
                closestEnemy = best;
        }

        return "tick=" + gs.getTime()
                + "\nmap=" + pgs.getWidth() + "x" + pgs.getHeight()
                + "\nresources=" + p.getResources()
                + "\nmy_workers=" + myWorkers
                + "\nmy_light=" + myLight
                + "\nmy_heavy=" + myHeavy
                + "\nmy_ranged=" + myRanged
                + "\nmy_barracks=" + myBarracks
                + "\nenemy_workers=" + enWorkers
                + "\nenemy_light=" + enLight
                + "\nenemy_heavy=" + enHeavy
                + "\nenemy_ranged=" + enRanged
                + "\nenemy_barracks=" + enBarracks
                + "\nclosest_enemy_to_base=" + closestEnemy
                + "\nprevious_strategy=" + current + "\n";
    }

    private Strategy parseStrategy(String text) {
        if (text == null)
            return current;
        String s = text.toUpperCase();
        if (s.contains("HEAVY"))
            return Strategy.HEAVY_RUSH;
        if (s.contains("RANGED"))
            return Strategy.RANGED_RUSH;
        if (s.contains("LIGHT"))
            return Strategy.LIGHT_RUSH;
        if (s.contains("WORKER"))
            return Strategy.WORKER_RUSH;
        return current;
    }

    private String callOllama(String stateBlock) {
        try {
            JsonObject options = new JsonObject();
            options.addProperty("temperature", 0);
            options.addProperty("seed", 0);

            JsonObject body = new JsonObject();
            body.addProperty("model", MODEL);
            body.addProperty("prompt", INSTRUCTION + stateBlock);
            body.addProperty("stream", false);
            body.add("options", options);

            URL url = new URL(OLLAMA_HOST + "/api/generate");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);

            try (OutputStream os = conn.getOutputStream()) {
                os.write(body.toString().getBytes(StandardCharsets.UTF_8));
            }

            int code = conn.getResponseCode();
            InputStream is = (code == 200) ? conn.getInputStream() : conn.getErrorStream();
            StringBuilder sb = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(is, StandardCharsets.UTF_8))) {
                for (String line; (line = br.readLine()) != null;)
                    sb.append(line);
            }

            if (code != 200) {
                System.err.println("[AI4PC] Ollama error " + code + ": " + sb);
                return null;
            }

            JsonObject top = JsonParser.parseString(sb.toString()).getAsJsonObject();
            return top.has("response") ? top.get("response").getAsString() : null;

        } catch (Exception e) {
            System.err.println("[AI4PC] " + e.getMessage());
            return null;
        }
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}
