package ai.mcts.submissions.xiebot;

import ai.RandomBiasedAI;
import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.evaluation.SimpleSqrtEvaluationFunction3;
import ai.mcts.naivemcts.NaiveMCTS;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

/**

XieBot v4

Practical tournament design for mixed map sizes:

Deterministic execution remains the safety net.
The LLM now chooses a bounded strategic plan that meaningfully affects economy,
production, posture, and target focus, especially on 16x16 and 32x32 maps.
MCTS is reserved for local tactical fights where lookahead is worth the cost.
*/
public class XieBot extends AbstractionLayerAI {
// ===== Core unit types =====
private UnitTypeTable utt;
private UnitType workerType;
private UnitType baseType;
private UnitType barracksType;
private UnitType lightType;
private UnitType rangedType;
private UnitType heavyType;

// ===== Safe macro modes (bounded strategy space) =====
private enum MacroPlan {
AGGRO_LIGHT, // Fast pressure, especially on 8x8
STABLE_RANGED, // Mix in ranged once stable
DEFENSIVE_HOLD // Survive rush first, then counter
}

private enum AttackPosture {
HOLD,
PROBE,
PRESSURE
}

private enum TargetFocus {
ECONOMY,
ARMY,
PRODUCTION,
BASE
}

// ===== LLM config =====
private static final int COMPACT_MAP_MAX_AREA = 100;
private static final int LARGE_MAP_MIN_AREA = 256;
private static final int HUGE_MAP_MIN_AREA = 900;
private static final String OLLAMA_ENDPOINT =
System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434") + "/api/generate";
private static final String OLLAMA_MODEL =
System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b");
private static final int LLM_CONNECT_TIMEOUT_MS = 120;
private static final int LLM_READ_TIMEOUT_MS = 260;
private static final int LLM_FAILURE_COOLDOWN = 4;

// ===== Tactical MCTS config =====
private final NaiveMCTS tacticalMCTS;
private static final int MCTS_TIME_BUDGET_MS = 35;
private static final int MCTS_LOOKAHEAD = 80;
private static final int MCTS_MAX_DEPTH = 8;
private static final int DEFENSE_RADIUS = 6;

// ===== Runtime state =====
private StrategicPlan strategicPlan =
new StrategicPlan(MacroPlan.AGGRO_LIGHT, AttackPosture.PROBE, TargetFocus.ECONOMY, 5, 1);
private int lastStrategicDecisionTime = -9999;
private int llmFailures = 0;

public XieBot(UnitTypeTable a_utt) {
super(new AStarPathFinding());
tacticalMCTS = new NaiveMCTS(
MCTS_TIME_BUDGET_MS,
-1,
MCTS_LOOKAHEAD,
MCTS_MAX_DEPTH,
0.25f,
0.0f,
0.35f,
new RandomBiasedAI(),
new SimpleSqrtEvaluationFunction3(),
true
);
reset(a_utt);
}

@Override
public void reset(UnitTypeTable a_utt) {
utt = a_utt;
workerType = utt.getUnitType("Worker");
baseType = utt.getUnitType("Base");
barracksType = utt.getUnitType("Barracks");
lightType = utt.getUnitType("Light");
rangedType = utt.getUnitType("Ranged");
heavyType = utt.getUnitType("Heavy");
reset();
}

@Override
public void reset() {
super.reset();
strategicPlan = new StrategicPlan(MacroPlan.AGGRO_LIGHT, AttackPosture.PROBE, TargetFocus.ECONOMY, 5, 1);
lastStrategicDecisionTime = -9999;
llmFailures = 0;
tacticalMCTS.reset();
}

@Override
public AI clone() {
return new XieBot(utt);
}

@Override
public PlayerAction getAction(int player, GameState gs) {
if (!gs.canExecuteAnyAction(player)) {
return new PlayerAction();
}

PhysicalGameState pgs = gs.getPhysicalGameState();
int enemy = 1 - player;
Player me = gs.getPlayer(player);
Player foe = gs.getPlayer(enemy);

Counts my = countUnits(pgs, player);
Counts opp = countUnits(pgs, enemy);

int mapArea = pgs.getWidth() * pgs.getHeight();
boolean compactMap = mapArea <= COMPACT_MAP_MAX_AREA;
boolean largeMap = mapArea >= LARGE_MAP_MIN_AREA;
int openingEnd = compactMap ? 320 : (largeMap ? 650 : 500);
boolean opening = gs.getTime() < openingEnd;
boolean enemyNearBase = isEnemyNearBase(pgs, my, enemy, largeMap ? 5 : 4);
int baseDistance = firstBaseDistance(my, opp);

maybeUpdateStrategicPlan(
gs,
my,
opp,
mapArea,
opening,
enemyNearBase,
baseDistance,
me.getResources(),
foe.getResources()
);

// Tactical MCTS is reserved for local, even fights rather than global navigation.
if (shouldUseTacticalMCTS(gs, my, opp, opening, enemyNearBase, mapArea, enemy)) {
    try {
        PlayerAction mctsAction = tacticalMCTS.getAction(player, gs);
        if (mctsAction != null && !mctsAction.isEmpty()) {
            return mctsAction;
        }
    } catch (Exception ignored) {
        // Fall back to the deterministic/LLM-guided policy immediately.
    }
}

// Deterministic execution of the current strategic plan:
// 1) production, 2) worker economy/defense, 3) military pressure.
Unit myBase = my.bases.isEmpty() ? null : my.bases.get(0);
Unit enemyBase = opp.bases.isEmpty() ? null : opp.bases.get(0);

handleBaseProduction(gs, player, my, enemyNearBase, mapArea, opening);
handleBarracksProduction(gs, player, my, opp, enemyNearBase, mapArea, opening);
handleWorkers(gs, player, my, opp, myBase, enemyBase, enemyNearBase, mapArea, opening);
handleMilitary(gs, enemy, my, opp, myBase, enemyBase, enemyNearBase, mapArea);

return translateActions(player, gs);
}

// ===========================
// Deterministic production
// ===========================
private void handleBaseProduction(GameState gs, int player, Counts my, boolean enemyNearBase, int mapArea, boolean opening) {
Player me = gs.getPlayer(player);

int desiredWorkers = strategicPlan.desiredWorkers;

// Under immediate threat with low army, keep worker count a little higher for emergency defense.
if (enemyNearBase && my.militaryCount() <= 1) {
    desiredWorkers = Math.max(desiredWorkers, 6);
}

if (opening && mapArea >= LARGE_MAP_MIN_AREA) {
    desiredWorkers = Math.max(desiredWorkers, 6);
}

desiredWorkers = clamp(desiredWorkers, 3, mapArea >= HUGE_MAP_MIN_AREA ? 10 : 8);

for (Unit base : my.bases) {
    if (!isIdle(gs, base)) continue;
    if (my.workers.size() < desiredWorkers && me.getResources() >= workerType.cost) {
        train(base, workerType);
        // Conservative accounting to avoid over-issuing train orders in same cycle.
        my.workers.add(new Unit(-1, workerType, 0, 0));
    }
}
}

private void handleBarracksProduction(
GameState gs,
int player,
Counts my,
Counts opp,
boolean enemyNearBase,
int mapArea,
boolean opening) {

Player me = gs.getPlayer(player);
for (Unit barracks : my.barracks) {
    if (!isIdle(gs, barracks)) continue;

    UnitType chosen = chooseBarracksUnit(my, opp, enemyNearBase, mapArea, opening);
    if (chosen != null && me.getResources() >= chosen.cost) {
        train(barracks, chosen);
    }
}
}

private UnitType chooseBarracksUnit(Counts my, Counts opp, boolean enemyNearBase, int mapArea, boolean opening) {
int oppCombatWithWorkers = opp.militaryCount() + opp.workers.size();

if (enemyNearBase && my.militaryCount() + 1 < oppCombatWithWorkers) {
    return lightType;
}

if (strategicPlan.macroPlan == MacroPlan.DEFENSIVE_HOLD) {
    if (my.light.size() < Math.max(2, my.ranged.size())) {
        return lightType;
    }
    return rangedType;
}

if (strategicPlan.macroPlan == MacroPlan.STABLE_RANGED) {
    if (opening && mapArea < LARGE_MAP_MIN_AREA) {
        return lightType;
    }
    if (my.ranged.size() <= my.light.size()) {
        return rangedType;
    }
    if (mapArea >= HUGE_MAP_MIN_AREA && my.heavy.size() * 3 < my.ranged.size() + 1) {
        return heavyType;
    }
    return lightType;
}

if (mapArea >= HUGE_MAP_MIN_AREA && !opening && my.heavy.size() < 1 && my.ranged.size() >= 2) {
    return heavyType;
}

if (strategicPlan.posture == AttackPosture.PRESSURE && my.light.size() <= my.ranged.size() + 1) {
    return lightType;
}

if (!opening && my.ranged.size() < my.light.size() / 2 && my.light.size() >= 4) {
    return rangedType;
}

return lightType;
}

// ===========================
// Worker manager (economy + anti-rush + build timing)
// ===========================
private void handleWorkers(
GameState gs,
int player,
Counts my,
Counts opp,
Unit myBase,
Unit enemyBase,
boolean enemyNearBase,
int mapArea,
boolean opening) {

Player me = gs.getPlayer(player);
int enemy = 1 - player;

List<Unit> idleWorkers = new ArrayList<>();
for (Unit w : my.workers) {
    if (isIdle(gs, w)) idleWorkers.add(w);
}

// Emergency anti-rush: if no army and enemy close, pull idle workers to fight now.
if (enemyNearBase && my.militaryCount() <= 1) {
    for (Unit w : idleWorkers) {
        Unit threat = nearestEnemy(gs.getPhysicalGameState(), w, enemy);
        if (threat != null) attack(w, threat);
    }
    return;
}

// Barracks timing: expand production more aggressively on 16x16/32x32 once economy is healthy.
if (shouldBuildBarracks(gs, my, opp, mapArea, opening, enemyNearBase)
        && me.getResources() >= barracksType.cost
        && !idleWorkers.isEmpty()) {
    Unit builder = selectBuilder(idleWorkers, myBase);
    int[] site = chooseBarracksSite(gs.getPhysicalGameState(), myBase != null ? myBase : builder, enemy);
    if (builder != null && site != null) {
        build(builder, barracksType, site[0], site[1]);
        idleWorkers.remove(builder);
    }
}

int harvestersWanted;
if (mapArea >= HUGE_MAP_MIN_AREA) {
    harvestersWanted = opening ? 5 : 4;
} else if (mapArea >= LARGE_MAP_MIN_AREA) {
    harvestersWanted = opening ? 4 : 3;
} else if (mapArea > COMPACT_MAP_MAX_AREA) {
    harvestersWanted = 3;
} else {
    harvestersWanted = 2;
}

if (strategicPlan.targetFocus == TargetFocus.ECONOMY || my.workers.size() < strategicPlan.desiredWorkers) {
    harvestersWanted++;
}
if (strategicPlan.posture == AttackPosture.PRESSURE && my.militaryCount() >= 6) {
    harvestersWanted--;
}
if (enemyNearBase) {
    harvestersWanted = Math.max(1, harvestersWanted - 1);
}
if (my.barracks.size() < strategicPlan.desiredBarracks && me.getResources() >= barracksType.cost) {
    harvestersWanted = Math.max(1, harvestersWanted - 1);
}

harvestersWanted = clamp(harvestersWanted, 1, mapArea >= HUGE_MAP_MIN_AREA ? 6 : 5);

// Keep at least one non-harvesting worker for flexibility.
harvestersWanted = Math.min(harvestersWanted, Math.max(0, my.workers.size() - 1));

List<Unit> resources = listResources(gs.getPhysicalGameState());
int assigned = 0;

for (Unit w : idleWorkers) {
    if (assigned >= harvestersWanted) break;
    Unit nearestRes = nearestResource(gs.getPhysicalGameState(), w, resources);
    if (nearestRes != null && myBase != null) {
        harvest(w, nearestRes, myBase);
        assigned++;
    }
}

boolean canRaidWithWorkers =
        strategicPlan.posture == AttackPosture.PRESSURE
        && my.militaryCount() >= 5
        && mapArea < HUGE_MAP_MIN_AREA;

// Remaining idle workers become local defenders or late-game raiders.
for (Unit w : idleWorkers) {
    if (!isIdle(gs, w)) continue;
    Unit target;
    if (enemyNearBase) {
        target = nearestEnemy(gs.getPhysicalGameState(), w, enemy);
    } else if (my.militaryCount() < 3 || myBase == null || strategicPlan.posture == AttackPosture.HOLD) {
        target = nearestEnemyWithin(gs.getPhysicalGameState(), w, enemy, mapArea >= LARGE_MAP_MIN_AREA ? DEFENSE_RADIUS + 2 : DEFENSE_RADIUS);
    } else if (canRaidWithWorkers && enemyBase != null) {
        target = enemyBase;
    } else {
        target = nearestEnemyWithin(gs.getPhysicalGameState(), w, enemy, mapArea >= LARGE_MAP_MIN_AREA ? DEFENSE_RADIUS + 2 : DEFENSE_RADIUS);
    }
    if (target != null) attack(w, target);
}
}

private boolean shouldBuildBarracks(GameState gs, Counts my, Counts opp, int mapArea, boolean opening, boolean enemyNearBase) {
int desiredBarracks = clamp(strategicPlan.desiredBarracks, 1, mapArea >= HUGE_MAP_MIN_AREA ? 3 : 2);

if (my.barracks.size() >= desiredBarracks) {
    return false;
}

if (my.barracks.isEmpty()) {
    return true;
}

if (enemyNearBase && my.militaryCount() < opp.militaryCount()) {
    return false;
}

Player p0 = gs.getPlayer(0);
Player p1 = gs.getPlayer(1);
int maxKnownResources = Math.max(p0.getResources(), p1.getResources());
int workerGate = mapArea >= LARGE_MAP_MIN_AREA ? 6 : 5;
int armyGate = opening ? (mapArea >= LARGE_MAP_MIN_AREA ? 2 : 3) : (mapArea >= HUGE_MAP_MIN_AREA ? 3 : 4);
int resourceGate = barracksType.cost + 2 + my.barracks.size() * 2;
return my.workers.size() >= workerGate && my.militaryCount() >= armyGate && maxKnownResources >= resourceGate;
}

// ===========================
// Military execution
// ===========================
private void handleMilitary(
GameState gs,
int enemy,
Counts my,
Counts opp,
Unit myBase,
Unit enemyBase,
boolean enemyNearBase,
int mapArea) {

List<Unit> army = new ArrayList<>();
army.addAll(my.light);
army.addAll(my.ranged);
army.addAll(my.heavy);

for (Unit unit : army) {
    if (!isIdle(gs, unit)) continue;

    Unit target = selectCombatTarget(gs.getPhysicalGameState(), unit, enemy, myBase, enemyNearBase, opp, mapArea);
    if (target == null && enemyBase != null && shouldAdvanceArmy(my, opp, mapArea, enemyNearBase)) {
        target = enemyBase;
    }
    if (target != null) {
        attack(unit, target);
    }
}
}

private Unit selectCombatTarget(PhysicalGameState pgs, Unit attacker, int enemy, Unit myBase, boolean urgentDefense, Counts opp, int mapArea) {
if (strategicPlan.posture == AttackPosture.HOLD && myBase != null && !urgentDefense) {
    Unit localThreat = nearestEnemyWithin(pgs, attacker, enemy, mapArea >= LARGE_MAP_MIN_AREA ? DEFENSE_RADIUS + 4 : DEFENSE_RADIUS + 2);
    if (localThreat != null) {
        return localThreat;
    }
}

List<Unit> enemies = new ArrayList<>();
for (Unit u : pgs.getUnits()) {
if (u.getPlayer() == enemy) enemies.add(u);
}
if (enemies.isEmpty()) return null;

enemies.sort(Comparator.comparingInt(e -> targetScore(attacker, e, myBase, urgentDefense, opp, mapArea)));
return enemies.get(0);
}

private int targetScore(Unit attacker, Unit target, Unit myBase, boolean urgentDefense, Counts opp, int mapArea) {
int d = manhattan(attacker, target);
int typeBias;

if (target.getType() == workerType) {
    // If enemy has low workers, each kill is very high value.
    typeBias = (opp.workers.size() <= 2) ? 1 : 5;
} else if (target.getType() == lightType || target.getType() == rangedType || target.getType() == heavyType) {
    typeBias = 3;
} else if (target.getType() == barracksType) {
    typeBias = 7;
} else if (target.getType() == baseType) {
    typeBias = 9;
} else {
    typeBias = 6;
}

if (strategicPlan.targetFocus == TargetFocus.ECONOMY && target.getType() == workerType) {
    typeBias -= 4;
} else if (strategicPlan.targetFocus == TargetFocus.ARMY && target.getType().canAttack) {
    typeBias -= 4;
} else if (strategicPlan.targetFocus == TargetFocus.PRODUCTION && target.getType() == barracksType) {
    typeBias -= 5;
} else if (strategicPlan.targetFocus == TargetFocus.BASE && target.getType() == baseType) {
    typeBias -= 5;
}

if (strategicPlan.posture != AttackPosture.PRESSURE
        && (target.getType() == baseType || target.getType() == barracksType)) {
    typeBias += 2;
}

if (urgentDefense && target.getType().canAttack) {
    typeBias -= 2;
}

if (attacker.getType() == rangedType && target.getType().canAttack) {
    typeBias -= 1;
}

int baseProximity = 0;
if (myBase != null) {
    baseProximity = manhattan(myBase, target);
}

int travelWeight = mapArea >= LARGE_MAP_MIN_AREA ? 2 : 3;
return d * travelWeight + typeBias + (urgentDefense ? baseProximity * 2 : 0);
}

// ===========================
// Strategic controller (heuristic first, bounded LLM plan)
// ===========================
private void maybeUpdateStrategicPlan(
GameState gs,
Counts my,
Counts opp,
int mapArea,
boolean opening,
boolean enemyNearBase,
int baseDistance,
int myResources,
int oppResources) {
if (gs.getTime() - lastStrategicDecisionTime < llmDecisionInterval(mapArea)) {
return;
}
lastStrategicDecisionTime = gs.getTime();

StrategicPlan heuristic = heuristicPlan(my, opp, mapArea, opening, enemyNearBase, baseDistance);

if (gs.getTime() < llmStartTick(mapArea)) {
    strategicPlan = heuristic;
    return;
}

// If the LLM has recently failed repeatedly, skip this cycle and cool down.
if (llmFailures >= LLM_FAILURE_COOLDOWN) {
    llmFailures--;
    strategicPlan = heuristic;
    return;
}

String llmRaw = queryStrategicLLM(
        gs.getTime(),
        my,
        opp,
        mapArea,
        opening,
        enemyNearBase,
        baseDistance,
        myResources,
        oppResources
);
StrategicPlan parsed = parseStrategicPlan(llmRaw, heuristic, mapArea);
if (parsed == null) {
    llmFailures++;
    strategicPlan = heuristic;
    return;
}

llmFailures = 0;
strategicPlan = parsed;
}

private StrategicPlan heuristicPlan(Counts my, Counts opp, int mapArea, boolean opening, boolean enemyNearBase, int baseDistance) {
int myArmy = my.militaryCount();
int oppArmy = opp.militaryCount();
boolean compactMap = mapArea <= COMPACT_MAP_MAX_AREA;
boolean largeMap = mapArea >= LARGE_MAP_MIN_AREA;

int desiredWorkers;
if (compactMap) {
    desiredWorkers = opening ? 4 : 5;
} else if (mapArea >= HUGE_MAP_MIN_AREA) {
    desiredWorkers = opening ? 7 : 8;
} else if (largeMap) {
    desiredWorkers = opening ? 6 : 7;
} else {
    desiredWorkers = opening ? 5 : 6;
}

int desiredBarracks = compactMap ? 1 : (mapArea >= HUGE_MAP_MIN_AREA ? 3 : 2);
if (opening) {
    desiredBarracks = 1;
}

MacroPlan macro = compactMap ? MacroPlan.AGGRO_LIGHT : MacroPlan.STABLE_RANGED;
AttackPosture posture = largeMap ? AttackPosture.PROBE : AttackPosture.PRESSURE;
TargetFocus focus = largeMap ? TargetFocus.ECONOMY : TargetFocus.PRODUCTION;

if (enemyNearBase || myArmy + 1 < oppArmy) {
    macro = MacroPlan.DEFENSIVE_HOLD;
    posture = AttackPosture.HOLD;
    focus = TargetFocus.ARMY;
    desiredWorkers = Math.max(desiredWorkers, 5);
} else if (largeMap && (opening || baseDistance >= 18 || my.workers.size() < desiredWorkers - 1)) {
    macro = MacroPlan.STABLE_RANGED;
    posture = AttackPosture.PROBE;
    focus = TargetFocus.ECONOMY;
} else if (!opening && my.ranged.size() >= my.light.size()) {
    macro = MacroPlan.STABLE_RANGED;
    posture = largeMap ? AttackPosture.PROBE : AttackPosture.PRESSURE;
    focus = opp.barracks.isEmpty() ? TargetFocus.BASE : TargetFocus.PRODUCTION;
} else if (myArmy > oppArmy + 2 && my.workers.size() >= desiredWorkers - 1) {
    macro = MacroPlan.AGGRO_LIGHT;
    posture = AttackPosture.PRESSURE;
    focus = opp.workers.size() > 2 ? TargetFocus.ECONOMY : TargetFocus.PRODUCTION;
}

desiredWorkers = clamp(desiredWorkers, 3, mapArea >= HUGE_MAP_MIN_AREA ? 10 : 8);
desiredBarracks = clamp(desiredBarracks, 1, mapArea >= HUGE_MAP_MIN_AREA ? 3 : 2);
return new StrategicPlan(macro, posture, focus, desiredWorkers, desiredBarracks);
}

private String queryStrategicLLM(
int time,
Counts my,
Counts opp,
int mapArea,
boolean opening,
boolean enemyNearBase,
int baseDistance,
int myResources,
int oppResources) {
String prompt =
"You are setting high-level MicroRTS strategy."
+ " Return exactly one line in this format and nothing else:"
+ " MACRO=<AGGRO_LIGHT|STABLE_RANGED|DEFENSIVE_HOLD>;"
+ "POSTURE=<HOLD|PROBE|PRESSURE>;"
+ "FOCUS=<ECONOMY|ARMY|PRODUCTION|BASE>;"
+ "WORKERS=<3-10>;"
+ "BARRACKS=<1-3>."
+ " On 16x16 and 32x32 maps, prioritize economy and stable ranged transitions before all-in pressure."
+ " Use HOLD only when behind or under direct threat."
+ " time=" + time
+ " mapArea=" + mapArea
+ " opening=" + opening
+ " enemyNearBase=" + enemyNearBase
+ " baseDistance=" + baseDistance
+ " myResources=" + myResources
+ " oppResources=" + oppResources
+ " my(workers=" + my.workers.size() + ",bases=" + my.bases.size() + ",barracks=" + my.barracks.size()
+ ",light=" + my.light.size() + ",ranged=" + my.ranged.size() + ",heavy=" + my.heavy.size() + ")"
+ " opp(workers=" + opp.workers.size() + ",bases=" + opp.bases.size() + ",barracks=" + opp.barracks.size()
+ ",light=" + opp.light.size() + ",ranged=" + opp.ranged.size() + ",heavy=" + opp.heavy.size() + ").";

HttpURLConnection conn = null;
try {
    URL url = new URL(OLLAMA_ENDPOINT);
    conn = (HttpURLConnection) url.openConnection();
    conn.setRequestMethod("POST");
    conn.setDoOutput(true);
    conn.setConnectTimeout(LLM_CONNECT_TIMEOUT_MS);
    conn.setReadTimeout(LLM_READ_TIMEOUT_MS);
    conn.setRequestProperty("Content-Type", "application/json");

    String body = "{"
            + "\"model\":\"" + jsonEscape(OLLAMA_MODEL) + "\","
            + "\"stream\":false,"
            + "\"prompt\":\"" + jsonEscape(prompt) + "\""
            + "}";

    try (OutputStream os = conn.getOutputStream()) {
        os.write(body.getBytes(StandardCharsets.UTF_8));
    }

    int code = conn.getResponseCode();
    if (code < 200 || code >= 300) return null;

    String response = readAll(conn.getInputStream());
    return extractResponseText(response);
} catch (Exception e) {
    return null;
} finally {
    if (conn != null) conn.disconnect();
}
}

private MacroPlan parseMacroToken(String text) {
String t = text == null ? "" : text.toUpperCase();
if (t.contains("AGGRO_LIGHT")) return MacroPlan.AGGRO_LIGHT;
if (t.contains("STABLE_RANGED")) return MacroPlan.STABLE_RANGED;
if (t.contains("DEFENSIVE_HOLD")) return MacroPlan.DEFENSIVE_HOLD;
return null;
}

private AttackPosture parsePostureToken(String text) {
String t = text == null ? "" : text.toUpperCase();
if (t.contains("POSTURE=HOLD") || t.contains(" HOLD")) return AttackPosture.HOLD;
if (t.contains("POSTURE=PROBE") || t.contains(" PROBE")) return AttackPosture.PROBE;
if (t.contains("POSTURE=PRESSURE") || t.contains(" PRESSURE")) return AttackPosture.PRESSURE;
return null;
}

private TargetFocus parseTargetFocusToken(String text) {
String t = text == null ? "" : text.toUpperCase();
if (t.contains("FOCUS=ECONOMY")) return TargetFocus.ECONOMY;
if (t.contains("FOCUS=ARMY")) return TargetFocus.ARMY;
if (t.contains("FOCUS=PRODUCTION")) return TargetFocus.PRODUCTION;
if (t.contains("FOCUS=BASE")) return TargetFocus.BASE;
return null;
}

private StrategicPlan parseStrategicPlan(String text, StrategicPlan fallback, int mapArea) {
if (text == null) return null;

MacroPlan macro = parseMacroToken(text);
AttackPosture posture = parsePostureToken(text);
TargetFocus focus = parseTargetFocusToken(text);
int desiredWorkers = parsePlanInt(text, "WORKERS=", fallback.desiredWorkers);
int desiredBarracks = parsePlanInt(text, "BARRACKS=", fallback.desiredBarracks);
boolean parsedAny = macro != null || posture != null || focus != null
        || text.toUpperCase().contains("WORKERS=") || text.toUpperCase().contains("BARRACKS=");
if (!parsedAny) return null;

if (macro == null) macro = fallback.macroPlan;
if (posture == null) posture = fallback.posture;
if (focus == null) focus = fallback.targetFocus;

desiredWorkers = clamp(desiredWorkers, 3, mapArea >= HUGE_MAP_MIN_AREA ? 10 : 8);
desiredBarracks = clamp(desiredBarracks, 1, mapArea >= HUGE_MAP_MIN_AREA ? 3 : 2);
return new StrategicPlan(macro, posture, focus, desiredWorkers, desiredBarracks);
}

// ===========================
// Tactical MCTS gating
// ===========================
private boolean shouldUseTacticalMCTS(
GameState gs,
Counts my,
Counts opp,
boolean opening,
boolean enemyNearBase,
int mapArea,
int enemy) {

boolean compactMap = mapArea <= COMPACT_MAP_MAX_AREA;
boolean largeMap = mapArea >= LARGE_MAP_MIN_AREA;

if (opening) return false; // deterministic openings are cheaper and safer.
if (enemyNearBase && my.militaryCount() <= 1) return false; // anti-rush worker logic should trigger.

int myArmy = my.militaryCount();
int oppArmy = opp.militaryCount();
if (myArmy < 4 || oppArmy < 4) return false;

// Use MCTS mainly when forces are comparable and already close to one another.
int diff = Math.abs(myArmy - oppArmy);
if (diff > 3) return false;
if (!hasLocalSkirmish(gs.getPhysicalGameState(), my, enemy, largeMap ? 4 : 5)) return false;

// On very small maps, keep MCTS usage conservative to avoid latency spikes.
if (compactMap && gs.getTime() < 700) return false;

// On large maps, let the strategic plan drive navigation; use MCTS only for local battles.
if (largeMap && strategicPlan.posture == AttackPosture.PRESSURE) return false;

// Do not invoke tactical MCTS when economy is fragile.
if (my.workers.size() < 2 || my.bases.isEmpty()) return false;

return true;
}

// ===========================
// Geometry/helpers
// ===========================
private boolean isIdle(GameState gs, Unit u) {
return gs.getActionAssignment(u) == null;
}

private boolean isEnemyNearBase(PhysicalGameState pgs, Counts my, int enemy, int radius) {
if (my.bases.isEmpty()) return false;
Unit base = my.bases.get(0);
for (Unit u : pgs.getUnits()) {
if (u.getPlayer() != enemy) continue;
if (manhattan(base, u) <= radius) return true;
}
return false;
}

private int[] chooseBarracksSite(PhysicalGameState pgs, Unit anchor, int enemy) {
int[][] ring = {
{1,0}, {-1,0}, {0,1}, {0,-1},
{2,0}, {-2,0}, {0,2}, {0,-2},
{1,1}, {-1,1}, {1,-1}, {-1,-1}
};

 int bestScore = Integer.MIN_VALUE;
 int[] best = null;

 for (int[] d : ring) {
     int x = anchor.getX() + d[0];
     int y = anchor.getY() + d[1];
     if (x < 0 || y < 0 || x >= pgs.getWidth() || y >= pgs.getHeight()) continue;
     if (pgs.getTerrain(x, y) != PhysicalGameState.TERRAIN_NONE) continue;
     if (pgs.getUnitAt(x, y) != null) continue;

     // Prefer sites that are not too far from base, and not too exposed to nearest enemy.
     int baseDist = Math.abs(anchor.getX() - x) + Math.abs(anchor.getY() - y);
     int enemyDist = nearestEnemyDistanceFromCell(pgs, x, y, enemy);
     int score = enemyDist * 2 - baseDist;

     if (score > bestScore) {
         bestScore = score;
         best = new int[] {x, y};
     }
 }
 return best;
}

private Unit selectBuilder(List<Unit> workers, Unit base) {
if (workers.isEmpty()) return null;
if (base == null) return workers.get(0);

 Unit best = null;
 int bestD = Integer.MAX_VALUE;
 for (Unit w : workers) {
     int d = manhattan(w, base);
     if (d < bestD) {
         bestD = d;
         best = w;
     }
 }
 return best;
}

private Unit nearestEnemy(PhysicalGameState pgs, Unit from, int enemy) {
Unit best = null;
int bestD = Integer.MAX_VALUE;
for (Unit u : pgs.getUnits()) {
if (u.getPlayer() != enemy) continue;
int d = manhattan(from, u);
if (d < bestD) {
bestD = d;
best = u;
}
}
return best;
}

private Unit nearestEnemyWithin(PhysicalGameState pgs, Unit from, int enemy, int maxDistance) {
 Unit best = null;
 int bestD = Integer.MAX_VALUE;
 for (Unit u : pgs.getUnits()) {
     if (u.getPlayer() != enemy) continue;
     int d = manhattan(from, u);
     if (d <= maxDistance && d < bestD) {
         bestD = d;
         best = u;
     }
 }
 return best;
}

private int nearestEnemyDistanceFromCell(PhysicalGameState pgs, int x, int y, int enemy) {
int best = Integer.MAX_VALUE;
for (Unit u : pgs.getUnits()) {
if (u.getPlayer() != enemy) continue;
int d = Math.abs(u.getX() - x) + Math.abs(u.getY() - y);
if (d < best) best = d;
}
return best;
}

private Unit nearestResource(PhysicalGameState pgs, Unit from, List<Unit> resources) {
Unit best = null;
int bestD = Integer.MAX_VALUE;
for (Unit r : resources) {
int d = manhattan(from, r);
if (d < bestD) {
bestD = d;
best = r;
}
}
return best;
}

private List<Unit> listResources(PhysicalGameState pgs) {
List<Unit> out = new ArrayList<>();
for (Unit u : pgs.getUnits()) {
if (u.getType().isResource) out.add(u);
}
return out;
}

private int manhattan(Unit a, Unit b) {
return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
}

private Counts countUnits(PhysicalGameState pgs, int player) {
Counts c = new Counts();
for (Unit u : pgs.getUnits()) {
if (u.getPlayer() != player) continue;
if (u.getType() == workerType) c.workers.add(u);
else if (u.getType() == baseType) c.bases.add(u);
else if (u.getType() == barracksType) c.barracks.add(u);
else if (u.getType() == lightType) c.light.add(u);
else if (u.getType() == rangedType) c.ranged.add(u);
else if (u.getType() == heavyType) c.heavy.add(u);
}
return c;
}

private int firstBaseDistance(Counts my, Counts opp) {
if (my.bases.isEmpty() || opp.bases.isEmpty()) {
return Integer.MAX_VALUE / 4;
}
return manhattan(my.bases.get(0), opp.bases.get(0));
}

private int llmDecisionInterval(int mapArea) {
if (mapArea >= HUGE_MAP_MIN_AREA) return 100;
if (mapArea >= LARGE_MAP_MIN_AREA) return 120;
if (mapArea > COMPACT_MAP_MAX_AREA) return 140;
return 160;
}

private int llmStartTick(int mapArea) {
if (mapArea >= HUGE_MAP_MIN_AREA) return 120;
if (mapArea >= LARGE_MAP_MIN_AREA) return 180;
if (mapArea > COMPACT_MAP_MAX_AREA) return 240;
return 320;
}

private int parsePlanInt(String text, String key, int fallback) {
String t = text == null ? "" : text.toUpperCase();
int idx = t.indexOf(key);
if (idx < 0) return fallback;

int start = idx + key.length();
while (start < t.length() && t.charAt(start) == ' ') {
start++;
}

int end = start;
while (end < t.length() && Character.isDigit(t.charAt(end))) {
end++;
}
if (end == start) return fallback;

try {
    return Integer.parseInt(t.substring(start, end));
} catch (NumberFormatException e) {
    return fallback;
}
}

private boolean shouldAdvanceArmy(Counts my, Counts opp, int mapArea, boolean enemyNearBase) {
if (enemyNearBase) return true;
if (strategicPlan.posture == AttackPosture.PRESSURE) return true;
if (strategicPlan.posture == AttackPosture.PROBE) {
    return my.militaryCount() >= (mapArea >= LARGE_MAP_MIN_AREA ? 5 : 4);
}
return my.militaryCount() > opp.militaryCount() + 2;
}

private boolean hasLocalSkirmish(PhysicalGameState pgs, Counts my, int enemy, int radius) {
List<Unit> army = new ArrayList<>();
army.addAll(my.light);
army.addAll(my.ranged);
army.addAll(my.heavy);
for (Unit ally : army) {
    for (Unit foe : pgs.getUnits()) {
        if (foe.getPlayer() != enemy) continue;
        if (!foe.getType().canAttack && foe.getType() != workerType) continue;
        if (manhattan(ally, foe) <= radius) return true;
    }
}
return false;
}

private int clamp(int value, int min, int max) {
return Math.max(min, Math.min(max, value));
}

// ===========================
// Lightweight HTTP/JSON helpers
// ===========================
private String readAll(InputStream in) throws IOException {
try (BufferedReader br = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
StringBuilder sb = new StringBuilder();
String line;
while ((line = br.readLine()) != null) sb.append(line);
return sb.toString();
}
}

private String extractResponseText(String json) {
if (json == null) return null;
int key = json.indexOf("\"response\"");
if (key < 0) return null;
int colon = json.indexOf(':', key);
if (colon < 0) return null;
int start = json.indexOf('"', colon + 1);
if (start < 0) return null;

 StringBuilder out = new StringBuilder();
 boolean escaped = false;
 for (int i = start + 1; i < json.length(); i++) {
     char ch = json.charAt(i);
     if (escaped) {
         out.append(ch);
         escaped = false;
         continue;
     }
     if (ch == '\\') {
         escaped = true;
         continue;
     }
     if (ch == '"') {
         break;
     }
     out.append(ch);
 }
 return out.toString().trim();
}

private String jsonEscape(String s) {
return s.replace("\\", "\\\\")
.replace("\"", "\\\"")
.replace("\n", " ");
}

private static final class StrategicPlan {
final MacroPlan macroPlan;
final AttackPosture posture;
final TargetFocus targetFocus;
final int desiredWorkers;
final int desiredBarracks;

 StrategicPlan(
         MacroPlan macroPlan,
         AttackPosture posture,
         TargetFocus targetFocus,
         int desiredWorkers,
         int desiredBarracks) {
     this.macroPlan = macroPlan;
     this.posture = posture;
     this.targetFocus = targetFocus;
     this.desiredWorkers = desiredWorkers;
     this.desiredBarracks = desiredBarracks;
 }
}

private static final class Counts {
final List<Unit> workers = new ArrayList<>();
final List<Unit> bases = new ArrayList<>();
final List<Unit> barracks = new ArrayList<>();
final List<Unit> light = new ArrayList<>();
final List<Unit> ranged = new ArrayList<>();
final List<Unit> heavy = new ArrayList<>();

 int militaryCount() {
     return light.size() + ranged.size() + heavy.size();
 }
}

@Override
public List<ParameterSpecification> getParameters() {
return new ArrayList<>();
}
}
