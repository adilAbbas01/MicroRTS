package ai.mcts.submissions.nick_mcts;

import ai.abstraction.LightRush;
import ai.abstraction.WorkerRush;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.RandomBiasedAI;
import ai.evaluation.LanchesterEvaluationFunction;
import ai.mcts.naivemcts.NaiveMCTS;
import java.util.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import com.google.gson.Gson;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import rts.GameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;
import rts.units.Unit;
import rts.PhysicalGameState;

public class NickMCTS extends NaiveMCTS {
    private UnitTypeTable utt;
    private StrategyController controller;
    private int lastUpdateFrame = -1;
    private boolean initialized = false;

    // Standard constructor for the game engine
    public NickMCTS(UnitTypeTable utt) {
        this(utt, new StrategyController());
    }

    // Shared-state constructor for cloning
    private NickMCTS(UnitTypeTable utt, StrategyController sc) {
        super(160, -1, 100, 25, 0.02f, 0.0f, 0.4f,
              new SafeLightRush(utt), 
              new MyEvaluation(utt, sc), 
              true);
        
        this.utt = utt;
        this.controller = sc;
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if (gs == null) return super.getAction(player, gs);

        // One-time check for map size to swap rollout policy
        if (!initialized) {
            if (gs.getPhysicalGameState().getWidth() <= 8) {
                this.playoutPolicy = new WorkerRush(utt);
            }
            initialized = true;
        }

        // Panic Trigger & LLM Strategy Update
        float currentThreat = ((MyEvaluation)this.ef).getLastThreat();
        boolean panicMode = currentThreat > 1.5f; 

        if (panicMode || (gs.getTime() % 200 == 0 && gs.getTime() != lastUpdateFrame)) {
            lastUpdateFrame = gs.getTime();
            boolean isSmall = gs.getPhysicalGameState().getWidth() <= 8;
            controller.updateStrategy(gs, player, isSmall, panicMode);
        }
        
        return super.getAction(player, gs);
    }

    @Override
    public AI clone() {
        // Correctly passing the controller reference to the clone
        return new NickMCTS(this.utt, this.controller);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}

class SafeLightRush extends LightRush {
    private AI fallback;
    public SafeLightRush(UnitTypeTable utt) {
        super(utt);
        this.fallback = new RandomBiasedAI();
    }
    @Override
    public PlayerAction getAction(int player, GameState gs) {
        try {
            return super.getAction(player, gs);
        } catch (Exception e) {
            try { return fallback.getAction(player, gs); } 
            catch (Exception e2) { return new PlayerAction(); }
        }
    }
    @Override
    public AI clone() { return new SafeLightRush(this.utt); }
}

class StrategyController {
    private static final String OLLAMA_HOST = System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    private static final String OLLAMA_MODEL = System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
    
    public volatile float aggression = 1.2f, threatWeight = 0.8f, resourceWeight = 0.15f, offensiveWeight = 1.0f; 

    private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofMillis(800)).build();
    private final Gson gson = new Gson();

    public void updateStrategy(GameState gs, int player, boolean isSmall, boolean panic) {
        String state = String.format("Map:%dx%d, Time:%d, Panic:%b, Res:%d", 
                       gs.getPhysicalGameState().getWidth(), gs.getPhysicalGameState().getHeight(), 
                       gs.getTime(), panic, gs.getPlayer(player).getResources());
        
        String prompt = "As a MicroRTS Master, analyze: " + state + ". Provide weights in JSON: " +
                        "{'analysis':'...', 'agg':0.5-3, 'thr':0-5, 'res':0-1, 'off':0-2}. JSON ONLY.";

        Map<String, Object> payload = new HashMap<>();
        payload.put("model", OLLAMA_MODEL);
        payload.put("prompt", prompt);
        payload.put("format", "json");
        payload.put("stream", false);
        
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(OLLAMA_HOST + "/api/generate"))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(gson.toJson(payload)))
                .build();

        client.sendAsync(request, HttpResponse.BodyHandlers.ofString())
              .thenApply(HttpResponse::body)
              .thenAccept(this::parseAndApply)
              .exceptionally(e -> null);
    }

    private void parseAndApply(String body) {
        try {
            JsonObject top = JsonParser.parseString(body).getAsJsonObject();
            JsonObject obj = JsonParser.parseString(top.get("response").getAsString()).getAsJsonObject();
            aggression = obj.get("agg").getAsFloat();
            threatWeight = obj.get("thr").getAsFloat();
            resourceWeight = obj.get("res").getAsFloat();
            offensiveWeight = obj.get("off").getAsFloat();
        } catch (Exception e) {}
    }
}

class MyEvaluation extends LanchesterEvaluationFunction {
    private StrategyController sc;
    private float lastThreat = 0;

    public MyEvaluation(UnitTypeTable utt, StrategyController sc) { this.sc = sc; }
    public void setController(StrategyController sc) { this.sc = sc; }
    public float getLastThreat() { return lastThreat; }

    @Override
    public float evaluate(int maxplayer, int minplayer, GameState gs) {
        float base = super.evaluate(maxplayer, minplayer, gs);
        
        float hpBonus = 0;
        for (Unit u : gs.getUnits()) {
            if (u.getPlayer() == maxplayer) {
                hpBonus += (u.getHitPoints() / (float)u.getMaxHitPoints()) * 0.1f;
            }
        }

        lastThreat = calculateThreat(maxplayer, gs, 0.05f);
        float score = (base * sc.aggression) - (lastThreat * sc.threatWeight) + hpBonus;
        return score + (calculateGlobalOffensiveBonus(maxplayer, gs) * sc.offensiveWeight);
    }

    private float calculateThreat(int player, GameState gs, float weight) {
        float p = 0;
        for (Unit u : gs.getUnits()) {
            if (u.getPlayer() == player && u.getType().name.equals("Base")) {
                for (Unit e : gs.getUnits()) {
                    if (e.getPlayer() == 1-player) {
                        int d = Math.abs(u.getX()-e.getX()) + Math.abs(u.getY()-e.getY());
                        if (d < 8) p += (8 - d) * weight;
                    }
                }
            }
        }
        return p;
    }

    private float calculateGlobalOffensiveBonus(int player, GameState gs) {
        Unit target = null;
        for (Unit u : gs.getUnits()) if (u.getPlayer() == 1-player && u.getType().name.equals("Base")) target = u;
        if (target == null) return 0;
        float b = 0;
        for (Unit u : gs.getUnits()) {
            if (u.getPlayer() == player && u.getType().canAttack) {
                int d = Math.abs(u.getX()-target.getX()) + Math.abs(u.getY()-target.getY());
                b += (20 - d) * 0.05f;
            }
        }
        return b;
    }
}

