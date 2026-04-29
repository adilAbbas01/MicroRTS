package ai.mcts.submissions.xiebot;

import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.UnitAction;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;

/**
 * XieBot
 *
 * A deliberately simple tournament bot.  The deterministic script owns every
 * concrete unit action; Ollama is only asked occasionally to choose a safe
 * macro preference among RUSH, BALANCED, and ECON.
 */
public class XieBot extends AbstractionLayerAI {
    private enum MacroMode {
        RUSH,
        BALANCED,
        ECON
    }

    private enum MapClass {
        SMALL,
        MEDIUM,
        LARGE
    }

    private static final boolean DEBUG = Boolean.getBoolean("xiebot.debug");

    private static final int LLM_OPENING_TICKS = 300;
    private static final int LLM_INTERVAL_TICKS = 320;
    private static final int LLM_CONNECT_TIMEOUT_MS = 100;
    private static final int LLM_READ_TIMEOUT_MS = 220;
    private static final int LLM_MAX_RESPONSE_CHARS = 4096;
    private static final String DEFAULT_OLLAMA_HOST = "http://127.0.0.1:11434";
    private static final String DEFAULT_OLLAMA_MODEL = "llama3.1:8b";

    private UnitTypeTable utt;
    private UnitType workerType;
    private UnitType baseType;
    private UnitType barracksType;
    private UnitType lightType;
    private UnitType rangedType;
    private UnitType heavyType;

    private MacroMode currentMode = MacroMode.BALANCED;
    private MacroMode lastLlmMode = null;
    private int lastLlmQueryTime = -9999;
    private int lastDebugTime = -9999;
    private int llmFailures = 0;

    public XieBot(UnitTypeTable a_utt) {
        super(new AStarPathFinding());
        reset(a_utt);
    }

    @Override
    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        if (utt != null) {
            workerType = utt.getUnitType("Worker");
            baseType = utt.getUnitType("Base");
            barracksType = utt.getUnitType("Barracks");
            lightType = utt.getUnitType("Light");
            rangedType = utt.getUnitType("Ranged");
            heavyType = utt.getUnitType("Heavy");
        }
        reset();
    }

    @Override
    public void reset() {
        super.reset();
        currentMode = MacroMode.BALANCED;
        lastLlmMode = null;
        lastLlmQueryTime = -9999;
        lastDebugTime = -9999;
        llmFailures = 0;
    }

    @Override
    public AI clone() {
        return new XieBot(utt);
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) {
        actions.clear();

        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);
        Snapshot s = analyze(player, gs, pgs);

        MacroMode heuristicMode = chooseHeuristicMode(s);
        MacroMode llmMode = maybeAskLlm(s, heuristicMode);
        currentMode = chooseFinalMode(s, heuristicMode, llmMode);

        int targetWorkers = targetWorkers(s, currentMode);
        int desiredBarracks = desiredBarracks(s, currentMode, targetWorkers);
        int availableResources = p.getResources();
        List<Integer> reservedPositions = new ArrayList<>();

        availableResources = maybeBuildReplacementBase(s, gs, pgs, p, availableResources, reservedPositions);
        availableResources = maybeBuildBarracks(s, gs, pgs, p, availableResources, reservedPositions, desiredBarracks, targetWorkers);
        availableResources = trainCombatUnits(s, gs, availableResources);
        availableResources = trainWorkers(s, gs, availableResources, targetWorkers);

        commandArmy(s, gs);
        commandWorkers(s, gs, pgs, p, reservedPositions);

        debugLog(s, heuristicMode, llmMode, currentMode, targetWorkers, desiredBarracks);
        return translateActions(player, gs);
    }

    private Snapshot analyze(int player, GameState gs, PhysicalGameState pgs) {
        Snapshot s = new Snapshot();
        s.player = player;
        s.enemy = 1 - player;
        s.time = gs.getTime();
        s.width = pgs.getWidth();
        s.height = pgs.getHeight();
        s.area = s.width * s.height;

        for (Unit u : pgs.getUnits()) {
            if (u.getType().isResource) {
                s.resources.add(u);
                continue;
            }

            if (u.getPlayer() == player) {
                addOwnUnit(s, u);
                UnitAction ua = gs.getUnitAction(u);
                if (ua != null && ua.getType() == UnitAction.TYPE_PRODUCE && ua.getUnitType() != null) {
                    countPendingOwnProduction(s, ua.getUnitType());
                }
            } else if (u.getPlayer() >= 0) {
                addEnemyUnit(s, u);
            }
        }

        s.mainBase = closestToCenter(s.ownBases, s.width, s.height);
        s.enemyBase = closestToCenter(s.enemyBases, s.width, s.height);
        s.mapClass = classifyMap(s);
        s.enemyThreats = findEnemyThreats(s);
        return s;
    }

    private void addOwnUnit(Snapshot s, Unit u) {
        if (u.getType() == workerType) {
            s.ownWorkers.add(u);
        } else if (u.getType() == baseType) {
            s.ownBases.add(u);
        } else if (u.getType() == barracksType) {
            s.ownBarracks.add(u);
        } else if (isMilitary(u)) {
            s.ownArmy.add(u);
            if (u.getType() == lightType) {
                s.ownLights++;
            } else if (u.getType() == rangedType) {
                s.ownRanged++;
            } else if (u.getType() == heavyType) {
                s.ownHeavy++;
            }
        }
    }

    private void addEnemyUnit(Snapshot s, Unit u) {
        s.enemyUnits.add(u);
        if (u.getType() == workerType) {
            s.enemyWorkers.add(u);
        } else if (u.getType() == baseType) {
            s.enemyBases.add(u);
        } else if (u.getType() == barracksType) {
            s.enemyBarracks.add(u);
        } else if (isMilitary(u)) {
            s.enemyArmy.add(u);
        }
    }

    private void countPendingOwnProduction(Snapshot s, UnitType type) {
        if (type == workerType) {
            s.pendingWorkers++;
        } else if (type == baseType) {
            s.pendingBases++;
        } else if (type == barracksType) {
            s.pendingBarracks++;
        } else if (type == lightType) {
            s.pendingLights++;
        } else if (type == rangedType) {
            s.pendingRanged++;
        }
    }

    private MapClass classifyMap(Snapshot s) {
        int baseDistance = distance(s.mainBase, s.enemyBase);
        if (s.area <= 144 || s.width <= 12 || s.height <= 12 || (baseDistance > 0 && baseDistance <= 16)) {
            return MapClass.SMALL;
        }
        if (s.area <= 576 || (baseDistance > 0 && baseDistance <= 32)) {
            return MapClass.MEDIUM;
        }
        return MapClass.LARGE;
    }

    private List<Unit> findEnemyThreats(Snapshot s) {
        List<Unit> threats = new ArrayList<>();
        for (Unit enemy : s.enemyUnits) {
            if (!enemy.getType().canAttack || !enemy.getType().canMove) {
                continue;
            }
            int radius;
            if (enemy.getType() == workerType) {
                radius = s.mapClass == MapClass.SMALL ? 2 : 3;
            } else {
                radius = s.mapClass == MapClass.SMALL ? 4 : (s.mapClass == MapClass.MEDIUM ? 6 : 8);
            }
            if (distanceToClosest(enemy, s.ownBases) <= radius || distanceToClosest(enemy, s.ownBarracks) <= radius) {
                threats.add(enemy);
            }
        }
        threats.sort(Comparator.comparingInt(u -> distanceToClosest(u, s.ownBases)));
        return threats;
    }

    private MacroMode chooseHeuristicMode(Snapshot s) {
        if (s.mapClass == MapClass.SMALL) {
            return MacroMode.RUSH;
        }
        if (s.isUnderThreat()) {
            return MacroMode.BALANCED;
        }
        if (hasArmyAdvantage(s)) {
            return MacroMode.RUSH;
        }
        if (s.mapClass == MapClass.LARGE && s.time < 1000 && s.totalWorkers() < 6) {
            return MacroMode.ECON;
        }
        return MacroMode.BALANCED;
    }

    private MacroMode chooseFinalMode(Snapshot s, MacroMode heuristicMode, MacroMode llmMode) {
        if (s.mapClass == MapClass.SMALL) {
            return MacroMode.RUSH;
        }
        if (s.isUnderThreat()) {
            return MacroMode.BALANCED;
        }
        if (hasArmyAdvantage(s)) {
            return MacroMode.RUSH;
        }
        if (llmMode == null) {
            return heuristicMode;
        }
        if (llmMode == MacroMode.ECON && s.enemyArmy.size() > s.ownArmy.size() + 1) {
            return MacroMode.BALANCED;
        }
        if (llmMode == MacroMode.RUSH && s.ownArmy.size() < 2 && s.time < 700) {
            return heuristicMode;
        }
        return llmMode;
    }

    private int targetWorkers(Snapshot s, MacroMode mode) {
        if (s.mapClass == MapClass.SMALL) {
            if (mode == MacroMode.RUSH) {
                if (s.time > 250 && s.enemyBarracks.isEmpty() && s.enemyArmy.isEmpty()) {
                    return 4;
                }
                return 1;
            }
            return 5;
        }
        if (s.mapClass == MapClass.MEDIUM) {
            return mode == MacroMode.ECON ? 6 : 5;
        }
        if (mode == MacroMode.RUSH) {
            return 5;
        }
        return mode == MacroMode.ECON ? 7 : 6;
    }

    private int desiredBarracks(Snapshot s, MacroMode mode, int targetWorkers) {
        if (s.mapClass == MapClass.LARGE
                && mode == MacroMode.ECON
                && s.time > 700
                && s.totalWorkers() >= targetWorkers) {
            return 2;
        }
        return 1;
    }

    private int maybeBuildReplacementBase(
            Snapshot s,
            GameState gs,
            PhysicalGameState pgs,
            Player p,
            int availableResources,
            List<Integer> reservedPositions) {
        if (s.totalBases() > 0 || availableResources < baseType.cost || s.ownWorkers.isEmpty()) {
            return availableResources;
        }

        Unit builder = closestIdleWorkerToAny(s.ownWorkers, s.resources, gs, false);
        if (builder == null) {
            return availableResources;
        }

        Unit anchor = closestUnit(builder, s.resources);
        if (anchor == null) {
            anchor = builder;
        }
        int pos = findBuildPosition(gs, pgs, anchor.getX(), anchor.getY(), reservedPositions);
        if (pos >= 0 && canCommand(builder, gs)) {
            build(builder, baseType, pos % pgs.getWidth(), pos / pgs.getWidth());
            reservedPositions.add(pos);
            return availableResources - baseType.cost;
        }
        return availableResources;
    }

    private int maybeBuildBarracks(
            Snapshot s,
            GameState gs,
            PhysicalGameState pgs,
            Player p,
            int availableResources,
            List<Integer> reservedPositions,
            int desiredBarracks,
            int targetWorkers) {
        if (s.totalBarracks() >= desiredBarracks || availableResources < barracksType.cost || s.ownWorkers.isEmpty()) {
            return availableResources;
        }

        boolean firstBarracks = s.totalBarracks() == 0;
        if (s.mapClass == MapClass.SMALL && !firstBarracks && s.isUnderThreat() && !s.enemyArmy.isEmpty()) {
            return availableResources;
        }

        int minWorkersBeforeFirstBarracks = s.mapClass == MapClass.SMALL ? 3 : 3;
        if (firstBarracks
                && s.mapClass != MapClass.SMALL
                && s.ownWorkers.size() < minWorkersBeforeFirstBarracks
                && s.time < 220) {
            return availableResources;
        }
        if (firstBarracks
                && s.mapClass != MapClass.SMALL
                && availableResources < barracksType.cost + lightType.cost
                && s.time < 220) {
            return availableResources;
        }
        if (!firstBarracks
                && (s.totalWorkers() < targetWorkers || availableResources < barracksType.cost + lightType.cost + workerType.cost)) {
            return availableResources;
        }

        Unit anchor = s.mainBase != null ? s.mainBase : s.ownWorkers.get(0);
        Unit builder = closestIdleWorkerTo(anchor, s.ownWorkers, gs, true);
        if (builder == null) {
            return availableResources;
        }

        int desiredX = anchor.getX();
        int desiredY = anchor.getY();
        if (s.enemyBase != null) {
            desiredX += Integer.compare(s.enemyBase.getX(), anchor.getX());
            desiredY += Integer.compare(s.enemyBase.getY(), anchor.getY());
        }

        int pos = findBuildPosition(gs, pgs, desiredX, desiredY, reservedPositions);
        if (pos >= 0 && canCommand(builder, gs)) {
            build(builder, barracksType, pos % pgs.getWidth(), pos / pgs.getWidth());
            reservedPositions.add(pos);
            return availableResources - barracksType.cost;
        }
        return availableResources;
    }

    private int trainCombatUnits(Snapshot s, GameState gs, int availableResources) {
        for (Unit barracks : s.ownBarracks) {
            if (!canCommand(barracks, gs) || !hasFreeAdjacent(barracks, gs)) {
                continue;
            }

            UnitType unitToTrain = chooseCombatUnit(s);
            if (availableResources >= unitToTrain.cost) {
                train(barracks, unitToTrain);
                availableResources -= unitToTrain.cost;
                if (unitToTrain == lightType) {
                    s.pendingLights++;
                } else if (unitToTrain == rangedType) {
                    s.pendingRanged++;
                }
            }
        }
        return availableResources;
    }

    private UnitType chooseCombatUnit(Snapshot s) {
        int lights = s.ownLights + s.pendingLights;
        int ranged = s.ownRanged + s.pendingRanged;
        boolean mixRanged = s.mapClass != MapClass.SMALL
                && currentMode != MacroMode.RUSH
                && !s.isUnderThreat()
                && lights >= 4
                && ranged * 3 < lights;
        return mixRanged ? rangedType : lightType;
    }

    private int trainWorkers(Snapshot s, GameState gs, int availableResources, int targetWorkers) {
        int workersPlanned = s.totalWorkers();
        if (workersPlanned >= targetWorkers) {
            return availableResources;
        }

        int minWorkersBeforeFirstBarracks = s.mapClass == MapClass.SMALL ? 3 : 3;
        boolean saveForFirstBarracks = s.totalBarracks() == 0
                && workersPlanned >= minWorkersBeforeFirstBarracks
                && availableResources < barracksType.cost + workerType.cost;
        if (saveForFirstBarracks) {
            return availableResources;
        }

        int plannedArmy = s.ownArmy.size() + s.pendingLights + s.pendingRanged;
        boolean saveForFirstLight = s.totalBarracks() > 0
                && workersPlanned >= 2
                && plannedArmy < (s.isUnderThreat() ? 2 : 1)
                && availableResources < lightType.cost + workerType.cost;
        if (saveForFirstLight) {
            return availableResources;
        }

        for (Unit base : s.ownBases) {
            if (workersPlanned >= targetWorkers || availableResources < workerType.cost) {
                break;
            }
            if (canCommand(base, gs) && hasFreeAdjacent(base, gs)) {
                train(base, workerType);
                availableResources -= workerType.cost;
                workersPlanned++;
                s.pendingWorkers++;
            }
        }
        return availableResources;
    }

    private void commandArmy(Snapshot s, GameState gs) {
        if (s.ownArmy.isEmpty()) {
            return;
        }

        if (s.isUnderThreat()) {
            for (Unit u : s.ownArmy) {
                Unit target = closestUnit(u, s.enemyThreats);
                if (target != null && canCommand(u, gs)) {
                    attack(u, target);
                }
            }
            return;
        }

        int threshold = attackThreshold(s, currentMode);
        boolean shouldAttack = s.ownArmy.size() >= threshold || hasArmyAdvantage(s);
        if (shouldAttack) {
            Unit target = chooseGlobalAttackTarget(s);
            if (target != null) {
                for (Unit u : s.ownArmy) {
                    if (canCommand(u, gs)) {
                        attack(u, target);
                    }
                }
                return;
            }
        }

        int[] rally = rallyPoint(s);
        for (Unit u : s.ownArmy) {
            if (canCommand(u, gs) && distance(u.getX(), u.getY(), rally[0], rally[1]) > 2) {
                move(u, rally[0], rally[1]);
            }
        }
    }

    private int attackThreshold(Snapshot s, MacroMode mode) {
        if (s.mapClass == MapClass.SMALL) {
            return mode == MacroMode.RUSH ? 1 : 2;
        }
        if (s.mapClass == MapClass.MEDIUM) {
            return mode == MacroMode.RUSH ? 3 : (mode == MacroMode.ECON ? 5 : 4);
        }
        return mode == MacroMode.RUSH ? 4 : (mode == MacroMode.ECON ? 6 : 5);
    }

    private Unit chooseGlobalAttackTarget(Snapshot s) {
        int[] center = armyCenter(s);
        Unit nearbyMilitary = closestWithin(center[0], center[1], s.enemyArmy, 8);
        if (nearbyMilitary != null) {
            return nearbyMilitary;
        }
        Unit worker = closestUnit(center[0], center[1], s.enemyWorkers);
        if (worker != null && (s.enemyBarracks.isEmpty() && s.enemyBases.isEmpty()
                || distance(center[0], center[1], worker.getX(), worker.getY()) <= 5
                || hasArmyAdvantage(s))) {
            return worker;
        }
        Unit barracks = closestUnit(center[0], center[1], s.enemyBarracks);
        if (barracks != null) {
            return barracks;
        }
        Unit base = closestUnit(center[0], center[1], s.enemyBases);
        if (base != null) {
            return base;
        }
        if (worker != null) {
            return worker;
        }
        return closestUnit(center[0], center[1], s.enemyUnits);
    }

    private void commandWorkers(
            Snapshot s,
            GameState gs,
            PhysicalGameState pgs,
            Player p,
            List<Integer> reservedPositions) {
        List<Unit> idleWorkers = new ArrayList<>();
        for (Unit worker : s.ownWorkers) {
            if (canCommand(worker, gs)) {
                idleWorkers.add(worker);
            }
        }

        pullDefendingWorkers(s, gs, idleWorkers);

        Unit closestBase = null;
        Unit closestResource = null;
        for (Unit worker : idleWorkers) {
            if (!canCommand(worker, gs)) {
                continue;
            }
            closestBase = closestUnit(worker, s.ownBases);
            closestResource = closestBase != null ? closestUnit(closestBase, s.resources) : closestUnit(worker, s.resources);
            if (closestBase != null && closestResource != null) {
                harvest(worker, closestResource, closestBase);
            } else if (s.ownArmy.size() >= s.enemyArmy.size() + 4 && !s.enemyUnits.isEmpty()) {
                Unit target = chooseGlobalAttackTarget(s);
                if (target != null) {
                    attack(worker, target);
                }
            }
        }
    }

    private void pullDefendingWorkers(Snapshot s, GameState gs, List<Unit> idleWorkers) {
        if (!s.isUnderThreat() || idleWorkers.isEmpty()) {
            return;
        }

        boolean severeThreat = hasSevereStructureThreat(s);
        int needed = s.enemyThreats.size() - s.ownArmy.size();
        if (needed <= 0 && !severeThreat) {
            return;
        }
        if (needed <= 0) {
            needed = 1;
        }
        boolean immediateSmallMapDefense = s.mapClass == MapClass.SMALL && s.ownArmy.isEmpty();
        int leaveHarvesters = immediateSmallMapDefense ? 0 : (idleWorkers.size() > 1 ? 1 : 0);
        int pullCount = Math.min(3, Math.min(needed, idleWorkers.size() - leaveHarvesters));
        if (pullCount <= 0) {
            return;
        }

        Unit anchor = s.mainBase != null ? s.mainBase : idleWorkers.get(0);
        int emptyHanded = 0;
        for (Unit worker : idleWorkers) {
            if (worker.getResources() == 0) {
                emptyHanded++;
            }
        }
        if (emptyHanded > 0) {
            pullCount = Math.min(pullCount, emptyHanded);
        }

        idleWorkers.sort(Comparator.comparingInt(w -> distance(w, anchor) + (w.getResources() > 0 ? 1000 : 0)));
        for (int i = 0; i < pullCount; i++) {
            Unit worker = idleWorkers.get(i);
            Unit target = closestUnit(worker, s.enemyThreats);
            if (target != null && canCommand(worker, gs)) {
                attack(worker, target);
            }
        }
    }

    private boolean hasSevereStructureThreat(Snapshot s) {
        for (Unit enemy : s.enemyThreats) {
            if (distanceToClosest(enemy, s.ownBases) <= 1 || distanceToClosest(enemy, s.ownBarracks) <= 1) {
                return true;
            }
        }
        return false;
    }

    private int[] rallyPoint(Snapshot s) {
        Unit anchor = s.mainBase != null ? s.mainBase : closestToCenter(s.ownWorkers, s.width, s.height);
        if (anchor == null) {
            return new int[] {s.width / 2, s.height / 2};
        }

        Unit target = s.enemyBase != null ? s.enemyBase : closestToCenter(s.enemyUnits, s.width, s.height);
        if (target == null) {
            return new int[] {anchor.getX(), anchor.getY()};
        }

        int x = anchor.getX();
        int y = anchor.getY();
        int steps = s.mapClass == MapClass.SMALL ? 2 : 4;
        for (int i = 0; i < steps; i++) {
            x += Integer.compare(target.getX(), x);
            y += Integer.compare(target.getY(), y);
        }
        return new int[] {clamp(x, 0, s.width - 1), clamp(y, 0, s.height - 1)};
    }

    private MacroMode maybeAskLlm(Snapshot s, MacroMode heuristicMode) {
        if (s.time < LLM_OPENING_TICKS || s.time - lastLlmQueryTime < LLM_INTERVAL_TICKS) {
            return lastLlmMode;
        }
        lastLlmQueryTime = s.time;

        String host = envOrDefault("OLLAMA_HOST", DEFAULT_OLLAMA_HOST);
        String model = envOrDefault("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL);
        try {
            String response = postOllama(host, model, buildPrompt(s, heuristicMode));
            MacroMode parsed = parseMacroMode(response);
            if (parsed != null) {
                lastLlmMode = parsed;
                return parsed;
            }
            llmFailures++;
        } catch (Exception e) {
            llmFailures++;
        }
        lastLlmMode = null;
        return null;
    }

    private String buildPrompt(Snapshot s, MacroMode heuristicMode) {
        return "MicroRTS macro choice. Reply with exactly one token: RUSH, BALANCED, or ECON.\n"
                + "Heuristic=" + heuristicMode
                + " map=" + s.mapClass
                + " time=" + s.time
                + " workers=" + s.ownWorkers.size()
                + " barracks=" + s.ownBarracks.size()
                + " army=" + s.ownArmy.size()
                + " enemyArmy=" + s.enemyArmy.size()
                + " enemyWorkers=" + s.enemyWorkers.size()
                + " enemyBarracks=" + s.enemyBarracks.size()
                + " baseThreat=" + s.isUnderThreat()
                + ".";
    }

    private String postOllama(String host, String model, String prompt) throws Exception {
        String normalizedHost = host.endsWith("/") ? host.substring(0, host.length() - 1) : host;
        URL url = new URL(normalizedHost + "/api/generate");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setConnectTimeout(LLM_CONNECT_TIMEOUT_MS);
        conn.setReadTimeout(LLM_READ_TIMEOUT_MS);
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json");

        String payload = "{\"model\":\"" + jsonEscape(model)
                + "\",\"prompt\":\"" + jsonEscape(prompt)
                + "\",\"stream\":false,\"options\":{\"temperature\":0,\"num_predict\":4}}";

        byte[] bytes = payload.getBytes(StandardCharsets.UTF_8);
        conn.setFixedLengthStreamingMode(bytes.length);
        try (OutputStream os = conn.getOutputStream()) {
            os.write(bytes);
        }

        InputStream stream = conn.getResponseCode() >= 400 ? conn.getErrorStream() : conn.getInputStream();
        if (stream == null) {
            return "";
        }
        return readLimited(stream, LLM_MAX_RESPONSE_CHARS);
    }

    private MacroMode parseMacroMode(String text) {
        if (text == null) {
            return null;
        }
        String upper = text.toUpperCase(Locale.ROOT);
        int rush = upper.indexOf("RUSH");
        int balanced = upper.indexOf("BALANCED");
        int econ = upper.indexOf("ECON");
        int best = Integer.MAX_VALUE;
        MacroMode mode = null;

        if (rush >= 0 && rush < best) {
            best = rush;
            mode = MacroMode.RUSH;
        }
        if (balanced >= 0 && balanced < best) {
            best = balanced;
            mode = MacroMode.BALANCED;
        }
        if (econ >= 0 && econ < best) {
            mode = MacroMode.ECON;
        }
        return mode;
    }

    private String readLimited(InputStream stream, int maxChars) throws Exception {
        StringBuilder sb = new StringBuilder();
        try (BufferedReader br = new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            int ch;
            while ((ch = br.read()) != -1 && sb.length() < maxChars) {
                sb.append((char) ch);
            }
        }
        return sb.toString();
    }

    private String jsonEscape(String value) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            switch (c) {
                case '\\':
                    sb.append("\\\\");
                    break;
                case '"':
                    sb.append("\\\"");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                default:
                    sb.append(c);
                    break;
            }
        }
        return sb.toString();
    }

    private String envOrDefault(String name, String fallback) {
        String value = System.getenv(name);
        return value == null || value.trim().isEmpty() ? fallback : value.trim();
    }

    private boolean canCommand(Unit u, GameState gs) {
        return u != null && gs.getActionAssignment(u) == null && !actions.containsKey(u);
    }

    private boolean hasFreeAdjacent(Unit u, GameState gs) {
        int[][] dirs = {{0, -1}, {1, 0}, {0, 1}, {-1, 0}};
        for (int[] d : dirs) {
            int x = u.getX() + d[0];
            int y = u.getY() + d[1];
            if (x >= 0 && y >= 0 && x < gs.getPhysicalGameState().getWidth() && y < gs.getPhysicalGameState().getHeight()
                    && gs.free(x, y)) {
                return true;
            }
        }
        return false;
    }

    private int findBuildPosition(GameState gs, PhysicalGameState pgs, int desiredX, int desiredY, List<Integer> reservedPositions) {
        int startX = clamp(desiredX, 0, pgs.getWidth() - 1);
        int startY = clamp(desiredY, 0, pgs.getHeight() - 1);
        int maxRadius = Math.max(pgs.getWidth(), pgs.getHeight());

        for (int r = 1; r <= maxRadius; r++) {
            int bestPos = -1;
            int bestScore = Integer.MAX_VALUE;
            for (int x = startX - r; x <= startX + r; x++) {
                for (int y = startY - r; y <= startY + r; y++) {
                    if (x < 0 || y < 0 || x >= pgs.getWidth() || y >= pgs.getHeight()) {
                        continue;
                    }
                    if (Math.abs(x - startX) != r && Math.abs(y - startY) != r) {
                        continue;
                    }
                    int pos = x + y * pgs.getWidth();
                    if (reservedPositions.contains(pos) || !gs.free(x, y)) {
                        continue;
                    }
                    int score = Math.abs(x - desiredX) + Math.abs(y - desiredY);
                    if (score < bestScore) {
                        bestScore = score;
                        bestPos = pos;
                    }
                }
            }
            if (bestPos >= 0) {
                return bestPos;
            }
        }
        return -1;
    }

    private boolean isMilitary(Unit u) {
        return u.getType().canAttack && u.getType().canMove && u.getType() != workerType;
    }

    private boolean hasArmyAdvantage(Snapshot s) {
        return s.ownArmy.size() >= 2 && s.ownArmy.size() >= s.enemyArmy.size() + 2;
    }

    private Unit closestIdleWorkerTo(Unit anchor, List<Unit> workers, GameState gs, boolean preferEmptyHands) {
        Unit best = null;
        int bestScore = Integer.MAX_VALUE;
        for (Unit worker : workers) {
            if (!canCommand(worker, gs)) {
                continue;
            }
            int carryPenalty = preferEmptyHands && worker.getResources() > 0 ? 1000 : 0;
            int score = distance(worker, anchor) + carryPenalty;
            if (score < bestScore) {
                bestScore = score;
                best = worker;
            }
        }
        return best;
    }

    private Unit closestIdleWorkerToAny(List<Unit> workers, List<Unit> anchors, GameState gs, boolean preferEmptyHands) {
        Unit best = null;
        int bestScore = Integer.MAX_VALUE;
        for (Unit worker : workers) {
            if (!canCommand(worker, gs)) {
                continue;
            }
            int carryPenalty = preferEmptyHands && worker.getResources() > 0 ? 1000 : 0;
            int score = distanceToClosest(worker, anchors) + carryPenalty;
            if (score < bestScore) {
                bestScore = score;
                best = worker;
            }
        }
        return best;
    }

    private Unit closestUnit(Unit from, List<Unit> units) {
        if (from == null || units == null || units.isEmpty()) {
            return null;
        }
        Unit best = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit u : units) {
            int d = distance(from, u);
            if (d < bestDist) {
                bestDist = d;
                best = u;
            }
        }
        return best;
    }

    private Unit closestUnit(int x, int y, List<Unit> units) {
        if (units == null || units.isEmpty()) {
            return null;
        }
        Unit best = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit u : units) {
            int d = distance(x, y, u.getX(), u.getY());
            if (d < bestDist) {
                bestDist = d;
                best = u;
            }
        }
        return best;
    }

    private Unit closestWithin(int x, int y, List<Unit> units, int maxDistance) {
        Unit best = closestUnit(x, y, units);
        return best != null && distance(x, y, best.getX(), best.getY()) <= maxDistance ? best : null;
    }

    private Unit closestToCenter(List<Unit> units, int width, int height) {
        if (units == null || units.isEmpty()) {
            return null;
        }
        int cx = width / 2;
        int cy = height / 2;
        Unit best = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit u : units) {
            int d = distance(u.getX(), u.getY(), cx, cy);
            if (d < bestDist) {
                bestDist = d;
                best = u;
            }
        }
        return best;
    }

    private int[] armyCenter(Snapshot s) {
        if (s.ownArmy.isEmpty()) {
            Unit anchor = s.mainBase != null ? s.mainBase : closestToCenter(s.ownWorkers, s.width, s.height);
            if (anchor != null) {
                return new int[] {anchor.getX(), anchor.getY()};
            }
            return new int[] {s.width / 2, s.height / 2};
        }
        int x = 0;
        int y = 0;
        for (Unit u : s.ownArmy) {
            x += u.getX();
            y += u.getY();
        }
        return new int[] {x / s.ownArmy.size(), y / s.ownArmy.size()};
    }

    private int distanceToClosest(Unit from, List<Unit> units) {
        if (from == null || units == null || units.isEmpty()) {
            return Integer.MAX_VALUE / 4;
        }
        int best = Integer.MAX_VALUE / 4;
        for (Unit u : units) {
            best = Math.min(best, distance(from, u));
        }
        return best;
    }

    private int distance(Unit a, Unit b) {
        if (a == null || b == null) {
            return Integer.MAX_VALUE / 4;
        }
        return distance(a.getX(), a.getY(), b.getX(), b.getY());
    }

    private int distance(int x1, int y1, int x2, int y2) {
        return Math.abs(x1 - x2) + Math.abs(y1 - y2);
    }

    private int clamp(int value, int min, int max) {
        return Math.max(min, Math.min(max, value));
    }

    private void debugLog(Snapshot s, MacroMode heuristicMode, MacroMode llmMode, MacroMode finalMode, int targetWorkers, int desiredBarracks) {
        if (!DEBUG || s.time - lastDebugTime < 80) {
            return;
        }
        lastDebugTime = s.time;
        System.out.println("[XieBot] t=" + s.time
                + " heuristic=" + heuristicMode
                + " llm=" + (llmMode == null ? "none" : llmMode)
                + " final=" + finalMode
                + " threat=" + s.isUnderThreat()
                + " workers=" + s.ownWorkers.size() + "/" + targetWorkers
                + " barracks=" + s.ownBarracks.size() + "/" + desiredBarracks
                + " army=" + s.ownArmy.size()
                + " enemyArmy=" + s.enemyArmy.size()
                + " llmFailures=" + llmFailures);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }

    private static final class Snapshot {
        int player;
        int enemy;
        int time;
        int width;
        int height;
        int area;
        MapClass mapClass = MapClass.MEDIUM;
        Unit mainBase;
        Unit enemyBase;
        int pendingWorkers;
        int pendingBases;
        int pendingBarracks;
        int pendingLights;
        int pendingRanged;
        int ownLights;
        int ownRanged;
        int ownHeavy;
        final List<Unit> ownBases = new ArrayList<>();
        final List<Unit> ownBarracks = new ArrayList<>();
        final List<Unit> ownWorkers = new ArrayList<>();
        final List<Unit> ownArmy = new ArrayList<>();
        final List<Unit> enemyBases = new ArrayList<>();
        final List<Unit> enemyBarracks = new ArrayList<>();
        final List<Unit> enemyWorkers = new ArrayList<>();
        final List<Unit> enemyArmy = new ArrayList<>();
        final List<Unit> enemyUnits = new ArrayList<>();
        final List<Unit> resources = new ArrayList<>();
        List<Unit> enemyThreats = new ArrayList<>();

        int totalWorkers() {
            return ownWorkers.size() + pendingWorkers;
        }

        int totalBases() {
            return ownBases.size() + pendingBases;
        }

        int totalBarracks() {
            return ownBarracks.size() + pendingBarracks;
        }

        boolean isUnderThreat() {
            return !enemyThreats.isEmpty();
        }
    }
}
