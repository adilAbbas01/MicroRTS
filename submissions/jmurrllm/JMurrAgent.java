package ai.abstraction.submissions.jmurrllm;

import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.WorkerRush;
import ai.abstraction.LightRush;
import ai.abstraction.HeavyRush;
import ai.abstraction.HybridLLMRush;
import ai.abstraction.RangedRush;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import rts.GameState;
import rts.PlayerAction;
import java.util.*;
import java.io.*;
import java.net.*;
import java.time.Instant;

import com.google.gson.*;
import rts.PhysicalGameState;
import rts.units.*;
import java.util.stream.Collectors;


public class JMurrAgent extends AbstractionLayerAI {

   protected UnitTypeTable utt;
   UnitType resourceType;
   UnitType workerType;
   UnitType lightType;
   UnitType heavyType;
   UnitType rangedType;
   UnitType baseType;
   UnitType barracksType;

   private WorkerRush workerRushAI;
   private LightRush lightRushAI;
   private HeavyRush heavyRushAI;
   private RangedRush rangedRushAI;

   private Strategy currentStrategy = Strategy.LIGHT_RUSH;

   public enum Strategy {
        WORKER_RUSH,
        LIGHT_RUSH,
        HEAVY_RUSH,
        RANGED_RUSH
   }

   public enum OppStrategy {
        WORKER_RUSH,
        LIGHT_RUSH,
        HEAVY_RUSH,
        RANGED_RUSH
   }

   public List<String> strategyNames = List.of(
        "worker rush",
        "light rush",
        "heavy rush",
        "ranged rush"
   );

   public Map<String, Strategy> counters = Map.of(
        "worker rush", Strategy.HEAVY_RUSH,
        "light rush", Strategy.RANGED_RUSH,
        "heavy rush", Strategy.RANGED_RUSH,
        "ranged rush", Strategy.WORKER_RUSH
    );

   private int lastLLMConsultation = -9999;
   private static final int START_TIME = 400;

   static final String OLLAMA_FORMAT = "json";
   static final Integer LLM_INTERVAL = 50;
   static final String ENDPOINT_URL =
        "https://generativelanguage.googleapis.com/v1beta/models/";
   static final String OLLAMA_HOST =
        System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    static String MODEL =
        System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
    
    static final boolean USE_CHAT = true;
    static final boolean OLLAMA_STREAM = false;

   public JMurrAgent(UnitTypeTable a_utt) {
      this(a_utt, new AStarPathFinding());
   }

   public JMurrAgent(UnitTypeTable a_utt, PathFinding a_pf) {
      super(a_pf);
      reset(a_utt);
   }

   @Override
   public void reset() {
      super.reset();
      if (workerRushAI != null) workerRushAI.reset();
      if (lightRushAI != null) lightRushAI.reset();
      if (heavyRushAI != null) heavyRushAI.reset();
      if (rangedRushAI != null) rangedRushAI.reset();
   }

   public void reset(UnitTypeTable a_utt) {
      utt = a_utt;
      resourceType = utt.getUnitType("Resource");
      workerType = utt.getUnitType("Worker");
      lightType = utt.getUnitType("Light");
      heavyType = utt.getUnitType("Heavy");
      rangedType = utt.getUnitType("Ranged");
      baseType = utt.getUnitType("Base");
      barracksType = utt.getUnitType("Barracks");
      
      workerRushAI = new WorkerRush(a_utt, pf) {
         private GameState _gs;
         @Override public PlayerAction getAction(int player, GameState gs) { _gs = gs; return super.getAction(player, gs); }
         @Override public void harvest(Unit u, Unit target, Unit base) { super.harvest(u, target != null ? target : closestResource(u), base); }
         private Unit closestResource(Unit u) {
            if (_gs == null) return null;
            Unit best = null; int bestD = Integer.MAX_VALUE;
            for (Unit u2 : _gs.getPhysicalGameState().getUnits()) {
               if (u2.getType().isResource) { int d = Math.abs(u2.getX()-u.getX())+Math.abs(u2.getY()-u.getY()); if (d < bestD) { best = u2; bestD = d; } }
            }
            return best;
         }
      };
      lightRushAI = new LightRush(a_utt, pf) {
         private GameState _gs;
         @Override public PlayerAction getAction(int player, GameState gs) { _gs = gs; return super.getAction(player, gs); }
         @Override public void harvest(Unit u, Unit target, Unit base) { super.harvest(u, target != null ? target : closestResource(u), base); }
         private Unit closestResource(Unit u) {
            if (_gs == null) return null;
            Unit best = null; int bestD = Integer.MAX_VALUE;
            for (Unit u2 : _gs.getPhysicalGameState().getUnits()) {
               if (u2.getType().isResource) { int d = Math.abs(u2.getX()-u.getX())+Math.abs(u2.getY()-u.getY()); if (d < bestD) { best = u2; bestD = d; } }
            }
            return best;
         }
      };
      heavyRushAI = new HeavyRush(a_utt, pf) {
         private GameState _gs;
         @Override public PlayerAction getAction(int player, GameState gs) { _gs = gs; return super.getAction(player, gs); }
         @Override public void harvest(Unit u, Unit target, Unit base) { super.harvest(u, target != null ? target : closestResource(u), base); }
         private Unit closestResource(Unit u) {
            if (_gs == null) return null;
            Unit best = null; int bestD = Integer.MAX_VALUE;
            for (Unit u2 : _gs.getPhysicalGameState().getUnits()) {
               if (u2.getType().isResource) { int d = Math.abs(u2.getX()-u.getX())+Math.abs(u2.getY()-u.getY()); if (d < bestD) { best = u2; bestD = d; } }
            }
            return best;
         }
      };
      rangedRushAI = new RangedRush(a_utt, pf) {
         private GameState _gs;
         @Override public PlayerAction getAction(int player, GameState gs) { _gs = gs; return super.getAction(player, gs); }
         @Override public void harvest(Unit u, Unit target, Unit base) { super.harvest(u, target != null ? target : closestResource(u), base); }
         private Unit closestResource(Unit u) {
            if (_gs == null) return null;
            Unit best = null; int bestD = Integer.MAX_VALUE;
            for (Unit u2 : _gs.getPhysicalGameState().getUnits()) {
               if (u2.getType().isResource) { int d = Math.abs(u2.getX()-u.getX())+Math.abs(u2.getY()-u.getY()); if (d < bestD) { best = u2; bestD = d; } }
            }
            return best;
         }
      };

      try {
        java.lang.reflect.Field f = ai.abstraction.AbstractionLayerAI.class.getDeclaredField("actions");
        f.setAccessible(true);
        f.set(workerRushAI, this.actions);
        f.set(lightRushAI,  this.actions);
        f.set(heavyRushAI,  this.actions);
        f.set(rangedRushAI, this.actions);
        System.out.println("Map sharing: " + (f.get(heavyRushAI) == this.actions ? "OK" : "FAILED"));
      } catch (Exception e) {
        System.err.println("Reflection failed: " + e);
        throw new RuntimeException("Failed to share actions map", e);
      }
   }

   // Start by prompting an LLM to mirror the opponent's strategy. Then, after a set number of ticks, switch to the HybridLLMRush AI.
   @Override
   public PlayerAction getAction(int player, GameState gs) throws Exception {
      int currentTime = gs.getTime();

      if (currentTime > START_TIME && (currentTime - lastLLMConsultation) >= LLM_INTERVAL) {
        String response = promptLLM(player, gs);
        parseResponse(response);

        lastLLMConsultation = currentTime;
      }
      
      switch (currentStrategy) {
        case WORKER_RUSH:
            return workerRushAI.getAction(player, gs);
        case LIGHT_RUSH:
            return lightRushAI.getAction(player, gs);
        case HEAVY_RUSH:
            return heavyRushAI.getAction(player, gs);
        case RANGED_RUSH:
            return rangedRushAI.getAction(player, gs);
        default:
            return workerRushAI.getAction(player, gs);
      }
   }

   @Override
   public AI clone() {
      JMurrAgent clone = new JMurrAgent(utt, pf);
      clone.currentStrategy = this.currentStrategy;
      return clone;
   }

   @Override
   public List<ParameterSpecification> getParameters() {
      List<ParameterSpecification> parameters = new ArrayList<>();
      parameters.add(new ParameterSpecification("PathFinding", PathFinding.class, new AStarPathFinding()));
      return parameters;
   }

   public String promptLLM(int player, GameState gs) {
      // ===== Gather game context =====
        PhysicalGameState pgs = gs.getPhysicalGameState();
        int width = pgs.getWidth();
        int height = pgs.getHeight();

        ArrayList<String> features = new ArrayList<>();

        // Build feature list for the prompt and help us count how many units we can legally command
        for (Unit u : pgs.getUnits()) {
            if(u.getPlayer() == player) {
                continue;
            }

            String unitType;
            if (u.getType() == resourceType) {
                unitType = "Resource Node";
            } else if (u.getType() == baseType) {
                unitType = "Base Unit";
            } else if (u.getType() == barracksType) {
                unitType = "Barracks Unit";
            } else if (u.getType() == workerType) {
                unitType = "Worker Unit";
            } else if (u.getType() == lightType) {
                unitType = "Light Unit";
            } else if (u.getType() == heavyType) {
                unitType = "Heavy Unit";
            } else if (u.getType() == rangedType) {
                unitType = "Ranged Unit";
            } else {
                unitType = "Unknown";
            }

            String unitPos = "(" + u.getX() + ", " + u.getY() + ")";
            String team = (u.getPlayer() == player) ? "Ally" :
                    (u.getType() == resourceType ? "Neutral" : "Enemy");

            features.add(unitPos + " " + team + " " + unitType);
        }
        
        String mapPrompt         = "Map size: " + width + "x" + height;
        String featuresPrompt = "Feature locations:\n" + String.join("\n", features);

        // Final LLM prompt
        String finalPrompt = PROMPT + "\n\n" +
                mapPrompt + "\n" +
                featuresPrompt + "\n";

        System.out.println("FINAL PROMPT: " + finalPrompt);

         // ===== Call the model (Ollama in your current version) =====
        String response = prompt(finalPrompt);
        // System.out.println("RESPONSE " + response);

        if (response instanceof String) {
            System.out.println("LLM returned String as expected.");
            // System.out.println(response);
            return response;
        }
        
        return null;
   }

   // This function was generated by Claude
   public static List<String> findInOrder(String text, List<String> targets) {
        String lowerText = text.toLowerCase();
        List<Map.Entry<String, Integer>> occurrences = new ArrayList<>();

        for (String target : targets) {
            String lowerTarget = target.toLowerCase();
            int index = 0;
            while ((index = lowerText.indexOf(lowerTarget, index)) != -1) {
                occurrences.add(new AbstractMap.SimpleEntry<>(target, index));
                index += target.length();
            }
        }

        occurrences.sort(Map.Entry.comparingByValue());

        return occurrences.stream()
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }

   public void parseResponse(String response) {
        List<String> strategyOccurences = findInOrder(response, strategyNames);
        String prediction = strategyOccurences.get(strategyOccurences.size()-1);
        System.out.println("Enemy strategy prediction: " + prediction);
        currentStrategy = counters.get(prediction);
   }

   public String prompt(String finalPrompt) {
        try {
            // Build Ollama request body
            JsonObject body = new JsonObject();
            body.addProperty("model", MODEL);
            // Prepend /no_think to disable qwen3 thinking mode for faster responses
            body.addProperty("prompt", "/no_think " + finalPrompt);
            body.addProperty("stream", OLLAMA_STREAM);   // false -> single JSON
            // body.addProperty("format", OLLAMA_FORMAT);   // "json" -> enforce JSON output


            URL url = new URL(OLLAMA_HOST + "/api/generate");
            HttpURLConnection conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("POST");
            conn.setRequestProperty("Content-Type", "application/json");
            conn.setDoOutput(true);


            try (OutputStream os = conn.getOutputStream()) {
                byte[] input = body.toString().getBytes(java.nio.charset.StandardCharsets.UTF_8);
                os.write(input);
            }

            int code = conn.getResponseCode();
            InputStream is = (code == HttpURLConnection.HTTP_OK)
                    ? conn.getInputStream()
                    : conn.getErrorStream();

            StringBuilder sb = new StringBuilder();
            try (BufferedReader br = new BufferedReader(new InputStreamReader(is, java.nio.charset.StandardCharsets.UTF_8))) {
                for (String line; (line = br.readLine()) != null; ) sb.append(line);
            }

            if (code != HttpURLConnection.HTTP_OK) {
                System.err.println("❌ Ollama error (" + code + "): " + sb);
                return "{\"thinking\":\"error\",\"moves\":[]}";
            }

            // Ollama /api/generate returns JSON like:
            // {"model":"...","created_at":"...","response":"...TEXT...","done":true,...}
            // Note: qwen3 "thinking" models put output in "thinking" field instead of "response"
            JsonObject top = JsonParser.parseString(sb.toString()).getAsJsonObject();

            String modelText = "";

            // First try "response" field (standard models like llama3.1)
            if (top.has("response") && !top.get("response").getAsString().isEmpty()) {
                modelText = top.get("response").getAsString();
            }
            // Fall back to "thinking" field (qwen3 thinking models)
            else if (top.has("thinking") && !top.get("thinking").isJsonNull()) {
                modelText = top.get("thinking").getAsString();
                System.out.println("📝 Using 'thinking' field from qwen3 model");
            }
            else {
                System.err.println("❌ Unexpected Ollama payload (no response or thinking): " + sb);
                return "{\"thinking\":\"invalid_response\",\"moves\":[]}";
            }

            // (Optional) log the raw text for debugging
            // System.out.println("OLLAMA raw response:\n" + modelText);

            // Return the text **as-is** — your caller will parse to JSON later
            return modelText;

        } catch (Exception e) {
            e.printStackTrace();
            return "{\"thinking\":\"exception\",\"moves\":[]}";
        }
    }

   String PROMPT = """
You are an AI playing a real-time strategy game. You control ALLY units only.
Given the following information, identify the opponent's strategy. Choose the best fitting opponent strategy from the following:
Worker rush: Opponent uses mainly Workers.
Light rush: Opponent uses mainly Light units.
Heavy rush: Opponent uses mainly Heavy units.
Ranged rush: Opponent uses mainly Ranged units.

In all cases the opponent will have one or two workers for harvesting. Other than those one or two workers, the opponent will only use units corresponding to their strategy.
""";
}