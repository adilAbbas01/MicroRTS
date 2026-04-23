package ai.mcts.submissions.penguin_bot;

import ai.abstraction.HeavyRush;
import ai.abstraction.RangedRush;
import ai.abstraction.WorkerDefense;
import ai.abstraction.WorkerRushPlusPlus;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.evaluation.SimpleSqrtEvaluationFunction3;
import ai.mcts.naivemcts.NaiveMCTS;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import rts.GameState;
import rts.PlayerAction;
import rts.ResourceUsage;
import rts.UnitAction;
import rts.UnitActionAssignment;
import rts.units.Unit;
import rts.units.UnitType;
import rts.units.UnitTypeTable;
import util.Pair;

/**
 * PenguinBot MCTS agent.
 *
 * <p>Architecture summary:
 *
 * <p>1) Opening book (deterministic macro build + WorkerDefense fill-ins)
 * <p>2) Deterministic strategic assessment (force ATTACK/DEFEND on clear triggers)
 * <p>3) Optional LLM strategy advice (stance + preferred unit)
 * <p>4) Stance-conditioned MCTS biasing and action selection
 * <p>5) Endgame finish mode to force cleanup and avoid stalling
 * <p>6) Runtime fallbacks to scripted policies if search fails
 */
public class MCTSAgent extends NaiveMCTS {

    private enum Stance {
        DEFEND,
        ATTACK
    }

    private enum Intent {
        OFFENSE,
        DEFENSE,
        ECONOMY,
        NEUTRAL
    }

    private static final int OPENING_END_TICK_DEFAULT = 360;
    private static final int OPENING_WORKERS_BEFORE_BARRACKS_DEFAULT = 1;
    private static final int OPENING_WORKER_TARGET_DEFAULT = 4;
    private static final int OPENING_RANGED_TARGET_DEFAULT = 1;
    private static final int OPENING_HEAVY_TARGET_DEFAULT = 1;
    private static final String WORKER_NAME = "Worker";
    private static final String BASE_NAME = "Base";
    private static final String BARRACKS_NAME = "Barracks";
    private static final String RANGED_NAME = "Ranged";
    private static final String HEAVY_NAME = "Heavy";

    private static final int RUSH_ALERT_RADIUS_DEFAULT = 7;
    private static final int BASE_DEFENSE_RADIUS_DEFAULT = 4;
    private static final int LARGE_MAP_AREA_THRESHOLD = 400;
    private static final int HEAVY_DEFENSE_TARGET = 2;
    private static final int LLM_INTERVAL = getEnvInt("MCTS_LLM_INTERVAL", 450);
    private static final boolean LLM_ENABLED =
            Boolean.parseBoolean(System.getenv().getOrDefault("MCTS_ENABLE_LLM", "true"));
    private static final int OLLAMA_CONNECT_TIMEOUT_MS =
            getEnvInt("OLLAMA_CONNECT_TIMEOUT_MS", 700);
    private static final int OLLAMA_READ_TIMEOUT_MS =
            getEnvInt("OLLAMA_READ_TIMEOUT_MS", 700);
    private static final int LLM_FAILURE_COOLDOWN_BASE_TICKS =
            getEnvInt("MCTS_LLM_FAILURE_COOLDOWN_BASE", 150);
    private static final int LLM_FAILURE_COOLDOWN_MAX_TICKS =
            getEnvInt("MCTS_LLM_FAILURE_COOLDOWN_MAX", 900);
    private static final String OLLAMA_HOST =
            System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");
    private static final String MODEL =
            System.getenv().getOrDefault("OLLAMA_MODEL", "gemma4");
    private static final String FALLBACK_MODELS_CSV =
            System.getenv().getOrDefault("OLLAMA_FALLBACK_MODELS", "llama3.2:3b,mistral:7b,qwen2.5:7b");

    private final UnitTypeTable utt;
    private final UnitType workerType;
    private final UnitType baseType;
    private final UnitType barracksType;
    private final UnitType rangedType;
    private final UnitType heavyType;
    private final HeavyRush heavyRushPolicy;
    private final RangedRush rangedRushPolicy;
    private final WorkerDefense workerDefensePolicy;
    private final WorkerRushPlusPlus workerRushPolicy;

    private int lastConsultTick = -9999;
    private int activePlayer = 0;
    private boolean openingComplete = false;
    private boolean finishMode = false;

    private Stance currentStance = Stance.DEFEND;
    private String preferredUnit = "RANGED";
    private String preferredReason = "Defensive opening";
    private Set<String> preferredActions = new HashSet<>();
    private String resolvedOllamaModel = null;
    private String llmPreferredUnitOverride = null;
    private boolean llmRangedAttackHeavyDefenseMode = false;
    private volatile boolean llmRequestInFlight = false;
    private volatile String pendingLlmResponse = null;
    private volatile int pendingLlmResponseTick = -1;
    private int llmFailureStreak = 0;
    private int llmCooldownUntilTick = -1;
    private int llmRequestGeneration = 0;

    /**
     * One-pass per-tick state summary used across opening, deterministic logic, and LLM prompting.
     */
    private static final class Snapshot {
        final GameState gs;
        final int player;
        final int enemy;
        final int time;
        final int mapW;
        final int mapH;
        final int mapArea;
        final int myResources;
        final int enemyResources;

        final List<Unit> myBases = new ArrayList<>();
        final List<Unit> myBarracksUnits = new ArrayList<>();
        final List<Unit> myWorkersUnits = new ArrayList<>();
        final List<Unit> enemyBases = new ArrayList<>();

        int myWorkers = 0;
        int myRanged = 0;
        int myHeavy = 0;
        int myBarracks = 0;
        int myCombat = 0;

        int enemyWorkers = 0;
        int enemyRanged = 0;
        int enemyHeavy = 0;
        int enemyBarracks = 0;
        int enemyCombat = 0;
        int enemyBaseCount = 0;

        Snapshot(GameState gs, int player) {
            this.gs = gs;
            this.player = player;
            this.enemy = 1 - player;
            this.time = gs.getTime();
            this.mapW = gs.getPhysicalGameState().getWidth();
            this.mapH = gs.getPhysicalGameState().getHeight();
            this.mapArea = mapW * mapH;
            this.myResources = gs.getPlayer(player).getResources();
            this.enemyResources = gs.getPlayer(enemy).getResources();
        }
    }

    /**
     * Mutable state for opening-book action construction.
     */
    private static final class OpeningContext {
        final Snapshot snapshot;
        final int player;
        final GameState gs;
        final List<Unit> myBases;
        final List<Unit> myBarracks;
        final List<Unit> myWorkers;
        final Map<Long, Pair<Unit, UnitAction>> defenseMap;
        final PlayerAction out;
        final Set<Long> assigned = new HashSet<>();
        int workerCount;
        int rangedCount;
        int heavyCount;

        OpeningContext(Snapshot snapshot, Map<Long, Pair<Unit, UnitAction>> defenseMap) {
            this.snapshot = snapshot;
            this.player = snapshot.player;
            this.gs = snapshot.gs;
            this.myBases = snapshot.myBases;
            this.myBarracks = snapshot.myBarracksUnits;
            this.myWorkers = snapshot.myWorkersUnits;
            this.workerCount = snapshot.myWorkers;
            this.rangedCount = snapshot.myRanged;
            this.heavyCount = snapshot.myHeavy;
            this.defenseMap = defenseMap;
            this.out = new PlayerAction();
            this.out.setResourceUsage(new ResourceUsage());
        }
    }

    public MCTSAgent(UnitTypeTable utt) {
        super(120, -1, 105, 10,
              0.30f, 0.0f, 0.40f,
              new RangedRush(utt),
              new SimpleSqrtEvaluationFunction3(),
              true);
        this.utt = utt;
        this.workerType = utt.getUnitType(WORKER_NAME);
        this.baseType = utt.getUnitType(BASE_NAME);
        this.barracksType = utt.getUnitType(BARRACKS_NAME);
        this.rangedType = utt.getUnitType(RANGED_NAME);
        this.heavyType = utt.getUnitType(HEAVY_NAME);
        this.heavyRushPolicy = new HeavyRush(utt);
        this.rangedRushPolicy = new RangedRush(utt);
        this.workerDefensePolicy = new WorkerDefense(utt);
        this.workerRushPolicy = new WorkerRushPlusPlus(utt);
        preferredActions.add("PRODUCE_HEAVY");
        preferredActions.add("PRODUCE_RANGED");
        preferredActions.add("DEFEND_BASE");
    }

    /**
     * Main decision pipeline for one game tick.
     *
     * <p>Order matters:
     *
     * <p>- tiny-map worker-rush shortcut
     * <p>- opening book (if not completed)
     * <p>- deterministic stance update
     * <p>- periodic LLM consult (unless finish mode is active)
     * <p>- stance bias application
     * <p>- finish-mode shortcut or MCTS action
     * <p>- scripted fallback on runtime errors
     */
    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if (!gs.canExecuteAnyAction(player)) return new PlayerAction();
        activePlayer = player;
        applyPendingOllamaResponse();

        if (isEightByEightMap(gs)) {
            openingComplete = true;
            currentStance = Stance.ATTACK;
            preferredReason = "Small map default: worker rush++";
            try {
                return workerRushPolicy.getAction(player, gs);
            } catch (RuntimeException ex) {
                return new PlayerAction();
            }
        }

        Snapshot snapshot = buildSnapshot(player, gs);

        if (!openingComplete && (snapshot.time < openingEndTick(snapshot.mapArea) || !openingGoalsMet(snapshot))) {
            return openingAction(snapshot);
        }
        openingComplete = true;

        boolean underAttack = isGettingRushed(snapshot);
        applyDeterministicStrategy(snapshot, underAttack);

        if (!finishMode) {
            scheduleOllamaConsult(snapshot, underAttack, false);
        }

        applyPendingOllamaResponse();
        applyDeterministicStrategy(snapshot, underAttack);
        applyStanceBiases();
        if (finishMode) {
            PlayerAction finisher = getFinishModeAction(snapshot);
            if (finisher != null && !finisher.isEmpty()) {
                return finisher;
            }
        }
        try {
            return super.getAction(player, gs);
        } catch (RuntimeException ex) {
            try {
                if (finishMode) {
                    PlayerAction finisher = getFinishModeAction(snapshot);
                    if (finisher != null) return finisher;
                }
                if (currentStance == Stance.ATTACK) {
                    return "HEAVY".equals(preferredUnit)
                            ? heavyRushPolicy.getAction(player, gs)
                            : rangedRushPolicy.getAction(player, gs);
                }
                return workerDefensePolicy.getAction(player, gs);
            } catch (RuntimeException ignored) {
                return new PlayerAction();
            }
        }
    }

    private Snapshot buildSnapshot(int player, GameState gs) {
        Snapshot snapshot = new Snapshot(gs, player);
        for (Unit unit : gs.getPhysicalGameState().getUnits()) {
            if (unit.getPlayer() == snapshot.player) {
                if (unit.getType() == baseType) {
                    snapshot.myBases.add(unit);
                } else if (unit.getType() == barracksType) {
                    snapshot.myBarracks++;
                    snapshot.myBarracksUnits.add(unit);
                } else if (unit.getType() == workerType) {
                    snapshot.myWorkers++;
                    snapshot.myWorkersUnits.add(unit);
                } else if (unit.getType() == rangedType) {
                    snapshot.myRanged++;
                } else if (unit.getType() == heavyType) {
                    snapshot.myHeavy++;
                }
                if (isCombatType(unit.getType())) snapshot.myCombat++;
            } else if (unit.getPlayer() == snapshot.enemy) {
                if (unit.getType() == baseType) {
                    snapshot.enemyBaseCount++;
                    snapshot.enemyBases.add(unit);
                } else if (unit.getType() == barracksType) {
                    snapshot.enemyBarracks++;
                } else if (unit.getType() == workerType) {
                    snapshot.enemyWorkers++;
                } else if (unit.getType() == rangedType) {
                    snapshot.enemyRanged++;
                } else if (unit.getType() == heavyType) {
                    snapshot.enemyHeavy++;
                }
                if (isCombatType(unit.getType())) snapshot.enemyCombat++;
            }
        }
        return snapshot;
    }

    /**
     * Deterministic opening book.
     *
     * <p>Policy goals:
     *
     * <p>- secure early workers
     * <p>- get first barracks safely
     * <p>- reach economy + seed military thresholds
     * <p>- borrow WorkerDefense for unassigned units while preventing production conflicts
     */
    private PlayerAction openingAction(Snapshot snapshot) throws Exception {
        currentStance = Stance.DEFEND;
        applyStanceBiases();

        OpeningContext ctx = createOpeningContext(snapshot);
        trainWorkerFromBaseIfNeeded(ctx, openingWorkersBeforeBarracks(ctx.gs));
        tryBuildFirstBarracks(ctx);
        trainWorkerFromBaseIfNeeded(ctx, openingWorkerTarget(ctx.gs));
        trainOpeningCombatUnits(ctx);
        mergeDefenseFallbackActions(ctx);

        if (openingGoalsMet(snapshot)) openingComplete = true;
        return ctx.out;
    }

    private OpeningContext createOpeningContext(Snapshot snapshot) throws Exception {
        PlayerAction defenseAction = workerDefensePolicy.getAction(snapshot.player, snapshot.gs);
        return new OpeningContext(snapshot, toActionMap(defenseAction));
    }

    private boolean isAssignedOrBusy(Unit unit, OpeningContext ctx) {
        return unit == null
                || ctx.gs.getActionAssignment(unit) != null
                || ctx.assigned.contains(unit.getID());
    }

    private void trainWorkerFromBaseIfNeeded(OpeningContext ctx, int targetCount) {
        if (ctx.workerCount >= targetCount) return;
        for (Unit base : ctx.myBases) {
            if (isAssignedOrBusy(base, ctx)) continue;
            if (!canAffordUnitTypeNow(ctx.player, workerType, ctx.out, ctx.gs)) break;
            UnitAction trainWorker = findProduceAction(base, workerType, ctx.gs);
            if (addIfConsistent(ctx.out, base, trainWorker, ctx.gs)) {
                ctx.assigned.add(base.getID());
                ctx.workerCount++;
                break;
            }
        }
    }

    private void tryBuildFirstBarracks(OpeningContext ctx) {
        boolean barracksReady = !ctx.myBarracks.isEmpty();
        boolean barracksInProgress = hasBarracksInProgress(ctx.player, ctx.gs);
        boolean baseIsProducing = hasBaseProduceInProgress(ctx.player, ctx.gs);
        if (barracksReady || barracksInProgress || baseIsProducing) return;

        Unit bestWorker = null;
        UnitAction bestBuild = null;
        int bestDist = Integer.MAX_VALUE;
        for (Unit worker : ctx.myWorkers) {
            if (isAssignedOrBusy(worker, ctx)) continue;
            if (!canAffordUnitTypeNow(ctx.player, barracksType, ctx.out, ctx.gs)) break;
            UnitAction buildBarracks = findProduceAction(worker, barracksType, ctx.gs);
            if (buildBarracks == null) continue;
            int d = distanceToClosest(worker, ctx.myBases);
            if (d < bestDist) {
                bestDist = d;
                bestWorker = worker;
                bestBuild = buildBarracks;
            }
        }
        if (bestWorker != null && addIfConsistent(ctx.out, bestWorker, bestBuild, ctx.gs)) {
            ctx.assigned.add(bestWorker.getID());
        }
    }

    private void trainOpeningCombatUnits(OpeningContext ctx) {
        if (ctx.myBarracks.isEmpty()) return;
        for (Unit barracks : ctx.myBarracks) {
            if (isAssignedOrBusy(barracks, ctx)) continue;
            if (ctx.rangedCount < openingRangedTarget(ctx.gs)) {
                if (tryTrainFromBarracks(ctx, barracks, rangedType)) ctx.rangedCount++;
                continue;
            }
            if (ctx.heavyCount < openingHeavyTarget(ctx.gs)) {
                if (tryTrainFromBarracks(ctx, barracks, heavyType)) ctx.heavyCount++;
                continue;
            }

            boolean preferRangedNow = ctx.rangedCount <= ctx.heavyCount;
            UnitType preferredType = preferRangedNow ? rangedType : heavyType;
            UnitType alternateType = preferRangedNow ? heavyType : rangedType;
            UnitAction preferred = findProduceAction(barracks, preferredType, ctx.gs);
            UnitAction alternate = findProduceAction(barracks, alternateType, ctx.gs);
            UnitAction chosen = preferred != null ? preferred : alternate;
            UnitType chosenType = preferred != null ? preferredType : alternateType;
            if (chosen == null) continue;
            if (!canAffordUnitTypeNow(ctx.player, chosenType, ctx.out, ctx.gs)) continue;
            if (addIfConsistent(ctx.out, barracks, chosen, ctx.gs)) {
                ctx.assigned.add(barracks.getID());
                if (chosen.getUnitType() == rangedType) ctx.rangedCount++;
                if (chosen.getUnitType() == heavyType) ctx.heavyCount++;
            }
        }
    }

    private boolean tryTrainFromBarracks(OpeningContext ctx, Unit barracks, UnitType type) {
        if (!canAffordUnitTypeNow(ctx.player, type, ctx.out, ctx.gs)) return false;
        UnitAction train = findProduceAction(barracks, type, ctx.gs);
        if (!addIfConsistent(ctx.out, barracks, train, ctx.gs)) return false;
        ctx.assigned.add(barracks.getID());
        return true;
    }

    private void mergeDefenseFallbackActions(OpeningContext ctx) {
        for (Unit unit : ctx.gs.getPhysicalGameState().getUnits()) {
            if (unit.getPlayer() != ctx.player || isAssignedOrBusy(unit, ctx)) continue;
            Pair<Unit, UnitAction> fromDefense = ctx.defenseMap.get(unit.getID());
            if (fromDefense == null || fromDefense.m_b == null) continue;
            if (fromDefense.m_b.getType() == UnitAction.TYPE_PRODUCE) continue;
            if (addIfConsistent(ctx.out, fromDefense.m_a, fromDefense.m_b, ctx.gs)) {
                ctx.assigned.add(unit.getID());
            }
        }
    }

    /**
     * Rule-based strategic override layer that sets global stance and preferred combat unit.
     *
     * <p>This runs before and after LLM consultation so deterministic safety constraints
     * remain authoritative when game-state triggers are decisive.
     */
    private void applyDeterministicStrategy(Snapshot snapshot, boolean underAttack) {
        int myWorkers = snapshot.myWorkers;
        int enemyWorkers = snapshot.enemyWorkers;
        int myRanged = snapshot.myRanged;
        int enemyRanged = snapshot.enemyRanged;
        int myHeavy = snapshot.myHeavy;
        int enemyHeavy = snapshot.enemyHeavy;
        int myBarracks = snapshot.myBarracks;
        int enemyBarracks = snapshot.enemyBarracks;
        int enemyBases = snapshot.enemyBaseCount;
        int myCombat = snapshot.myCombat;
        int enemyCombat = snapshot.enemyCombat;

        updateFinishMode(snapshot.player, snapshot.gs,
                myCombat, enemyCombat, myWorkers, enemyWorkers, enemyBarracks, enemyBases);
        if (finishMode) {
            currentStance = Stance.ATTACK;
            preferredReason = "Finish mode: force aggressive cleanup after decisive lead";
        }

        boolean enemyCollapsed = enemyBarracks == 0 && enemyCombat == 0 && enemyWorkers <= 1;
        int leadThreshold = snapshot.mapArea <= 256 ? 3 : 4;
        boolean strongCombatLead = myCombat >= leadThreshold && myCombat >= enemyCombat + 2;
        boolean structuralLead = myBarracks > 0 && enemyBarracks == 0 && myCombat >= Math.max(2, enemyCombat + 1);
        boolean economySnowball = myWorkers >= enemyWorkers + 6 && myCombat >= enemyCombat;
        boolean forceAttack = enemyCollapsed || strongCombatLead || structuralLead || economySnowball;
        boolean forceDefend = underAttack && !forceAttack && myCombat <= enemyCombat;

        if (finishMode || forceAttack) {
            currentStance = Stance.ATTACK;
            preferredReason = "Deterministic attack trigger from decisive board advantage";
        } else if (forceDefend) {
            currentStance = Stance.DEFEND;
            preferredReason = "Deterministic defend trigger while under pressure";
        }

        if (llmRangedAttackHeavyDefenseMode
                && snapshot.mapArea >= LARGE_MAP_AREA_THRESHOLD
                && !forceDefend) {
            currentStance = Stance.ATTACK;
            preferredReason = "LLM large-map split mode: ranged pressure with heavy base anchors";
        }

        String deterministicUnit;
        if (enemyRanged > enemyHeavy + 1) {
            deterministicUnit = "HEAVY";
        } else if (enemyHeavy > enemyRanged + 1) {
            deterministicUnit = "RANGED";
        } else {
            deterministicUnit = myRanged <= myHeavy ? "RANGED" : "HEAVY";
        }

        if (isLlmPreferredUnitActive()) {
            preferredUnit = llmPreferredUnitOverride;
        } else {
            llmPreferredUnitOverride = null;
            preferredUnit = deterministicUnit;
        }
        if (llmRangedAttackHeavyDefenseMode && snapshot.mapArea >= LARGE_MAP_AREA_THRESHOLD) {
            preferredUnit = "RANGED";
        }
    }

    /**
     * Latches (and clears) finish mode.
     *
     * <p>Finish mode is entered when the opponent has no meaningful army production/pressure left
     * but still has remaining units/buildings to clean up.
     */
    private void updateFinishMode(int player, GameState gs, int myCombat, int enemyCombat,
                                  int myWorkers, int enemyWorkers, int enemyBarracks, int enemyBases) {
        boolean enemyArmyGone = enemyBarracks == 0 && enemyCombat == 0;
        boolean enemyStillAlive = enemyWorkers > 0 || enemyBases > 0;
        boolean militaryCleanupReady = myCombat >= Math.max(1, enemyCombat + 1);
        boolean workerCleanupReady = myCombat == 0 && myWorkers >= Math.max(3, enemyWorkers + 2);

        if (!finishMode && enemyArmyGone && enemyStillAlive && (militaryCleanupReady || workerCleanupReady)) {
            finishMode = true;
            return;
        }

        if (finishMode && (!enemyStillAlive || !gs.canExecuteAnyAction(player))) {
            finishMode = false;
            return;
        }

        if (finishMode && enemyBarracks > 0 && enemyCombat > myCombat) {
            finishMode = false;
        }
    }

    /**
     * Returns a direct scripted cleanup policy used during finish mode.
     */
    private PlayerAction getFinishModeAction(Snapshot snapshot) {
        int player = snapshot.player;
        GameState gs = snapshot.gs;
        try {
            if (snapshot.myCombat > 0) {
                return "HEAVY".equals(preferredUnit)
                        ? heavyRushPolicy.getAction(player, gs)
                        : rangedRushPolicy.getAction(player, gs);
            }
            return workerRushPolicy.getAction(player, gs);
        } catch (RuntimeException ex) {
            return new PlayerAction();
        }
    }

    private UnitAction findProduceAction(Unit unit, UnitType unitType, GameState gs) {
        if (unit == null || unitType == null) return null;
        for (UnitAction ua : unit.getUnitActions(gs)) {
            if (ua.getType() == UnitAction.TYPE_PRODUCE && ua.getUnitType() == unitType) {
                return ua;
            }
        }
        return null;
    }

    private boolean hasBarracksInProgress(int player, GameState gs) {
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && u.getType() == barracksType) return true;
        }
        for (UnitActionAssignment uaa : gs.getUnitActions().values()) {
            if (uaa.unit.getPlayer() == player
                    && uaa.action.getType() == UnitAction.TYPE_PRODUCE
                    && uaa.action.getUnitType() == barracksType) {
                return true;
            }
        }
        return false;
    }

    private boolean hasBaseProduceInProgress(int player, GameState gs) {
        for (UnitActionAssignment uaa : gs.getUnitActions().values()) {
            if (uaa == null || uaa.unit == null || uaa.action == null) continue;
            if (uaa.unit.getPlayer() != player) continue;
            if (uaa.unit.getType() != baseType) continue;
            if (uaa.action.getType() == UnitAction.TYPE_PRODUCE) return true;
        }
        return false;
    }

    private void scheduleOllamaConsult(Snapshot snapshot, boolean underAttack, boolean force) {
        if (!LLM_ENABLED || finishMode) return;

        final int requestTick = snapshot.time;
        int consultInterval = underAttack ? Math.max(10, LLM_INTERVAL / 4) : LLM_INTERVAL;
        final int generation;
        synchronized (this) {
            if (!force && requestTick < llmCooldownUntilTick) return;
            if (!force && requestTick - lastConsultTick < consultInterval) return;
            if (llmRequestInFlight) return;
            llmRequestInFlight = true;
            generation = llmRequestGeneration;
            lastConsultTick = requestTick;
        }

        final String prompt;
        try {
            prompt = buildPrompt(snapshot, underAttack);
        } catch (RuntimeException ex) {
            synchronized (this) {
                if (llmRequestGeneration == generation) {
                    recordLlmFailureLocked(requestTick);
                    llmRequestInFlight = false;
                }
            }
            return;
        }

        Thread worker = new Thread(() -> {
            String response = null;
            try {
                response = callOllama(prompt);
            } catch (Exception ignored) {
            } finally {
                synchronized (MCTSAgent.this) {
                    if (llmRequestGeneration != generation) return;
                    if (response != null && !response.trim().isEmpty()) {
                        pendingLlmResponse = response;
                        pendingLlmResponseTick = requestTick;
                        recordLlmSuccessLocked();
                    } else {
                        recordLlmFailureLocked(requestTick);
                    }
                    llmRequestInFlight = false;
                }
            }
        }, "penguinbot-ollama");
        worker.setDaemon(true);
        worker.start();
    }

    private void applyPendingOllamaResponse() {
        String raw;
        int tick;
        synchronized (this) {
            if (pendingLlmResponse == null) return;
            raw = pendingLlmResponse;
            tick = pendingLlmResponseTick;
            pendingLlmResponse = null;
            pendingLlmResponseTick = -1;
        }
        boolean applied = parseStrategyFromResponse(raw);
        synchronized (this) {
            if (applied) {
                recordLlmSuccessLocked();
            } else {
                recordLlmFailureLocked(Math.max(0, tick));
            }
        }
    }

    /**
     * Builds the strategy prompt for Ollama.
     *
     * <p>The LLM is asked for high-level control only (not low-level unit actions).
     */
    private String buildPrompt(Snapshot snapshot, boolean underAttack) {
        int myPressure = minMyCombatDistanceToEnemyBase(snapshot.player, snapshot.gs, snapshot.enemyBases);
        int enemyPressure = minEnemyCombatDistanceToMyBase(snapshot.player, snapshot.gs, snapshot.myBases);
        if (myPressure == Integer.MAX_VALUE) myPressure = 999;
        if (enemyPressure == Integer.MAX_VALUE) enemyPressure = 999;

        boolean largeMap = snapshot.mapArea >= LARGE_MAP_AREA_THRESHOLD;
        String suggested = largeMap
                ? "RANGED"
                : (snapshot.enemyRanged > snapshot.enemyHeavy ? "HEAVY" : "RANGED");
        String example = buildPromptExample(suggested, largeMap);

        StringBuilder sb = new StringBuilder();
        appendPromptRules(sb);
        appendPromptState(sb, snapshot, underAttack, myPressure, enemyPressure);
        appendPromptSchema(sb);
        sb.append("Example: ").append(example);
        return sb.toString();
    }

    private String buildPromptExample(String suggestedUnit, boolean largeMap) {
        return "{\"switch_required\":false,\"target_stance\":\"" + currentStance.name()
                + "\",\"necessity\":\"NOT_NECESSARY\",\"preferred_unit\":\"" + suggestedUnit
                + "\",\"ranged_attack_heavy_defense\":" + (largeMap ? "true" : "false")
                + ",\"reason\":\"hold stance and do not split behavior\"}";
    }

    private void appendPromptRules(StringBuilder sb) {
        sb.append("You are a strict stance controller for an RTS bot. Return JSON only.\n");
        sb.append("The bot has binary stances only: DEFEND or ATTACK.\n");
        sb.append("Do not suggest mixed behavior except when ranged_attack_heavy_defense=true.\n");
        sb.append("If map is 16x16, prioritize a fast military build over long-term macro.\n");
        sb.append("On larger maps (map_area >= 400), prefer preferred_unit=RANGED in most cases.\n");
        sb.append("Only pick preferred_unit=HEAVY on larger maps when enemy ranged pressure clearly demands it.\n");
        sb.append("On larger maps, prefer ranged_attack_heavy_defense=true: send ranged units forward to attack while keeping about 2 heavy units near own base on defense.\n");
        sb.append("Switch only when it is wholly necessary. If not wholly necessary, keep stance unchanged.\n");
    }

    private void appendPromptState(StringBuilder sb, Snapshot snapshot, boolean underAttack,
                                   int myPressure, int enemyPressure) {
        sb.append("State:\n");
        sb.append("- map: ").append(snapshot.mapW).append("x").append(snapshot.mapH).append("\n");
        sb.append("- map_area: ").append(snapshot.mapArea).append("\n");
        sb.append("- time: ").append(snapshot.time).append("\n");
        sb.append("- current_stance: ").append(currentStance.name()).append("\n");
        sb.append("- under_attack: ").append(underAttack).append("\n");
        sb.append("- my_resources: ").append(snapshot.myResources).append("\n");
        sb.append("- enemy_resources: ").append(snapshot.enemyResources).append("\n");
        sb.append("- my_workers: ").append(snapshot.myWorkers).append("\n");
        sb.append("- my_heavy: ").append(snapshot.myHeavy).append("\n");
        sb.append("- my_ranged: ").append(snapshot.myRanged).append("\n");
        sb.append("- my_barracks: ").append(snapshot.myBarracks).append("\n");
        sb.append("- my_combat_units: ").append(snapshot.myCombat).append("\n");
        sb.append("- enemy_workers: ").append(snapshot.enemyWorkers).append("\n");
        sb.append("- enemy_heavy: ").append(snapshot.enemyHeavy).append("\n");
        sb.append("- enemy_ranged: ").append(snapshot.enemyRanged).append("\n");
        sb.append("- enemy_barracks: ").append(snapshot.enemyBarracks).append("\n");
        sb.append("- enemy_combat_units: ").append(snapshot.enemyCombat).append("\n");
        sb.append("- my_frontline_to_enemy_base: ").append(myPressure).append("\n");
        sb.append("- enemy_frontline_to_my_base: ").append(enemyPressure).append("\n");
    }

    private void appendPromptSchema(StringBuilder sb) {
        sb.append("JSON schema:\n");
        sb.append("{\"switch_required\":true|false,");
        sb.append("\"target_stance\":\"DEFEND|ATTACK\",");
        sb.append("\"necessity\":\"WHOLLY_NECESSARY|NOT_NECESSARY\",");
        sb.append("\"preferred_unit\":\"HEAVY|RANGED\",");
        sb.append("\"ranged_attack_heavy_defense\":true|false,");
        sb.append("\"reason\":\"short explanation\",");
        sb.append("\"wholly_necessary\":true|false(optional)}\n");
    }

    private String callOllama(String prompt) throws Exception {
        String model = getResolvedOllamaModel();
        try {
            return callOllamaWithModel(model, prompt);
        } catch (Exception e) {
            String fallback = pickFallbackModel(model);
            if (fallback.equals(model)) throw e;
            resolvedOllamaModel = fallback;
            return callOllamaWithModel(fallback, prompt);
        }
    }

    private String callOllamaWithModel(String model, String prompt) throws Exception {
        URL url = new URL(OLLAMA_HOST + "/api/generate");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setRequestMethod("POST");
        conn.setConnectTimeout(OLLAMA_CONNECT_TIMEOUT_MS);
        conn.setReadTimeout(OLLAMA_READ_TIMEOUT_MS);
        conn.setDoOutput(true);
        conn.setRequestProperty("Content-Type", "application/json");

        JsonObject body = new JsonObject();
        body.addProperty("model", model);
        body.addProperty("prompt", prompt);
        body.addProperty("stream", false);
        body.addProperty("format", "json");

        try (OutputStream os = conn.getOutputStream()) {
            os.write(body.toString().getBytes(StandardCharsets.UTF_8));
        }

        int status = conn.getResponseCode();
        InputStream stream = status >= 200 && status < 300 ? conn.getInputStream() : conn.getErrorStream();
        if (stream == null) throw new RuntimeException("Ollama HTTP " + status + " with empty response");

        byte[] raw;
        try (InputStream is = stream) {
            raw = readFully(is);
        }
        String envelope = new String(raw, StandardCharsets.UTF_8);
        if (status < 200 || status >= 300) {
            throw new RuntimeException("Ollama HTTP " + status + ": " + envelope);
        }
        JsonObject root = JsonParser.parseString(envelope).getAsJsonObject();
        return root.has("response") ? root.get("response").getAsString() : envelope;
    }

    private String getResolvedOllamaModel() {
        if (resolvedOllamaModel != null && !resolvedOllamaModel.isEmpty()) return resolvedOllamaModel;

        List<String> installed = listInstalledOllamaModels();
        if (installed.isEmpty()) {
            resolvedOllamaModel = MODEL;
            return resolvedOllamaModel;
        }

        for (String candidate : getModelCandidates()) {
            String found = findInstalledModelName(installed, candidate);
            if (found != null) {
                resolvedOllamaModel = found;
                return resolvedOllamaModel;
            }
        }

        resolvedOllamaModel = installed.get(0);
        return resolvedOllamaModel;
    }

    private String pickFallbackModel(String failedModel) {
        List<String> installed = listInstalledOllamaModels();
        if (installed.isEmpty()) return failedModel;

        for (String candidate : getModelCandidates()) {
            if (candidate.equalsIgnoreCase(failedModel)) continue;
            String found = findInstalledModelName(installed, candidate);
            if (found != null) return found;
        }

        for (String installedModel : installed) {
            if (!installedModel.equalsIgnoreCase(failedModel)) return installedModel;
        }
        return failedModel;
    }

    private List<String> getModelCandidates() {
        LinkedHashSet<String> out = new LinkedHashSet<>();
        if (MODEL != null && !MODEL.trim().isEmpty()) out.add(MODEL.trim());
        for (String raw : FALLBACK_MODELS_CSV.split(",")) {
            String m = raw.trim();
            if (!m.isEmpty()) out.add(m);
        }
        return new ArrayList<>(out);
    }

    private String findInstalledModelName(List<String> installed, String candidate) {
        if (candidate == null || candidate.trim().isEmpty()) return null;
        for (String modelName : installed) {
            if (candidate.equalsIgnoreCase(modelName)) return modelName;
        }
        return null;
    }

    private List<String> listInstalledOllamaModels() {
        List<String> models = new ArrayList<>();
        HttpURLConnection conn = null;
        try {
            URL url = new URL(OLLAMA_HOST + "/api/tags");
            conn = (HttpURLConnection) url.openConnection();
            conn.setRequestMethod("GET");
            conn.setConnectTimeout(1200);
            conn.setReadTimeout(1800);

            int status = conn.getResponseCode();
            if (status < 200 || status >= 300) return models;

            try (InputStream is = conn.getInputStream()) {
                JsonObject root = JsonParser.parseString(new String(readFully(is), StandardCharsets.UTF_8)).getAsJsonObject();
                if (!root.has("models") || !root.get("models").isJsonArray()) return models;
                for (JsonElement e : root.getAsJsonArray("models")) {
                    if (!e.isJsonObject()) continue;
                    JsonObject obj = e.getAsJsonObject();
                    if (!obj.has("name")) continue;
                    String name = obj.get("name").getAsString();
                    if (name != null && !name.trim().isEmpty()) models.add(name.trim());
                }
            }
        } catch (Exception ignored) {
        } finally {
            if (conn != null) conn.disconnect();
        }
        return models;
    }

    private byte[] readFully(InputStream is) throws Exception {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        byte[] buffer = new byte[4096];
        int n;
        while ((n = is.read(buffer)) != -1) {
            out.write(buffer, 0, n);
        }
        return out.toByteArray();
    }

    /**
     * Parses and applies strategic signals from LLM output.
     *
     * <p>Accepted signals:
     *
     * <p>- stance switch (only if marked wholly necessary)
     * <p>- preferred combat unit
     */
    private boolean parseStrategyFromResponse(String raw) {
        JsonObject strategy = parseStrategyJson(raw);
        if (strategy == null) return false;

        Stance target = currentStance;
        if (strategy.has("target_stance")) {
            String v = strategy.get("target_stance").getAsString().toUpperCase();
            if ("ATTACK".equals(v)) target = Stance.ATTACK;
            if ("DEFEND".equals(v)) target = Stance.DEFEND;
        } else if (strategy.has("stance")) {
            String v = strategy.get("stance").getAsString().toUpperCase();
            if ("ATTACK".equals(v)) target = Stance.ATTACK;
            if ("DEFEND".equals(v)) target = Stance.DEFEND;
        }

        boolean switchRequired = strategy.has("switch_required") && strategy.get("switch_required").getAsBoolean();
        boolean whollyNecessary = false;
        if (strategy.has("necessity")) {
            String necessity = strategy.get("necessity").getAsString().toUpperCase();
            whollyNecessary = necessity.contains("WHOLLY");
        }
        if (strategy.has("wholly_necessary")) {
            whollyNecessary = whollyNecessary || strategy.get("wholly_necessary").getAsBoolean();
        }
        if (switchRequired && whollyNecessary && target != currentStance) {
            currentStance = target;
        }

        if (strategy.has("preferred_unit")) {
            String v = strategy.get("preferred_unit").getAsString().toUpperCase();
            if ("HEAVY".equals(v) || "RANGED".equals(v)) {
                llmPreferredUnitOverride = v;
                preferredUnit = v;
            }
        }
        if (strategy.has("ranged_attack_heavy_defense")) {
            llmRangedAttackHeavyDefenseMode = strategy.get("ranged_attack_heavy_defense").getAsBoolean();
        } else if (strategy.has("formation")) {
            String v = strategy.get("formation").getAsString().toUpperCase();
            llmRangedAttackHeavyDefenseMode = v.contains("RANGED_ATTACK_HEAVY_DEFENSE");
        }
        if (strategy.has("reason")) preferredReason = strategy.get("reason").getAsString();
        return true;
    }

    private JsonObject parseStrategyJson(String raw) {
        String trimmed = raw == null ? "" : raw.trim();
        if (trimmed.isEmpty()) return null;

        try {
            JsonElement direct = JsonParser.parseString(trimmed);
            if (direct.isJsonObject()) return direct.getAsJsonObject();
        } catch (Exception ignored) {
        }

        String jsonObject = extractFirstJsonObject(trimmed);
        if (jsonObject == null) return null;
        try {
            JsonElement extracted = JsonParser.parseString(jsonObject);
            if (extracted.isJsonObject()) return extracted.getAsJsonObject();
        } catch (Exception ignored) {
        }
        return null;
    }

    private String extractFirstJsonObject(String text) {
        int start = -1;
        int depth = 0;
        boolean inString = false;
        boolean escaped = false;

        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (escaped) {
                escaped = false;
                continue;
            }
            if (c == '\\') {
                escaped = true;
                continue;
            }
            if (c == '"') {
                inString = !inString;
                continue;
            }
            if (inString) continue;
            if (c == '{') {
                if (depth == 0) start = i;
                depth++;
                continue;
            }
            if (c == '}' && depth > 0) {
                depth--;
                if (depth == 0 && start >= 0) {
                    return text.substring(start, i + 1);
                }
            }
        }
        return null;
    }

    /**
     * Applies stance-dependent MCTS parameterization and preferred action categories.
     */
    private void applyStanceBiases() {
        preferredActions.clear();

        if (finishMode) {
            MAXSIMULATIONTIME = 140;
            MAX_TREE_DEPTH = 10;
            initial_epsilon_0 = 0.62f;
            initial_epsilon_l = 0.25f;
            initial_epsilon_g = 0.0f;
            playoutPolicy = "HEAVY".equals(preferredUnit) ? heavyRushPolicy : rangedRushPolicy;
            Collections.addAll(preferredActions,
                    "ATTACK_NEAR_BASE",
                    "ADVANCE",
                    "PRODUCE_RANGED",
                    "PRODUCE_HEAVY",
                    "PRODUCE_" + preferredUnit);
            return;
        }

        if (currentStance == Stance.DEFEND) {
            MAXSIMULATIONTIME = 110;
            MAX_TREE_DEPTH = 13;
            initial_epsilon_0 = 0.16f;
            initial_epsilon_l = 0.54f;
            initial_epsilon_g = 0.0f;
            playoutPolicy = workerDefensePolicy;
            Collections.addAll(preferredActions,
                    "HARVEST",
                    "RETURN",
                    "BUILD_BARRACKS",
                    "PRODUCE_WORKER",
                    "PRODUCE_RANGED",
                    "PRODUCE_HEAVY",
                    "DEFEND_BASE");
            return;
        }

        MAXSIMULATIONTIME = 130;
        MAX_TREE_DEPTH = 11;
        initial_epsilon_0 = 0.58f;
        initial_epsilon_l = 0.28f;
        initial_epsilon_g = 0.0f;
        playoutPolicy = isRangedAttackHeavyDefenseModeActive() ? rangedRushPolicy
                : ("HEAVY".equals(preferredUnit) ? heavyRushPolicy : rangedRushPolicy);
        Collections.addAll(preferredActions,
                "ATTACK_NEAR_BASE",
                "ADVANCE",
                "PRODUCE_RANGED",
                "PRODUCE_HEAVY",
                "PRODUCE_" + preferredUnit,
                "PRODUCE_WORKER");
        if (isRangedAttackHeavyDefenseModeActive()) {
            preferredActions.add("DEFEND_BASE");
        }
    }

    /**
     * Selects the final action from MCTS children using a combined score:
     * visit count + preference score + average evaluation.
     */
    @Override
    public int getMostVisitedActionIdx() {
        total_actions_issued++;
        if (tree == null || tree.children == null || tree.children.isEmpty()) return -1;

        int bestIdx = -1;
        double bestScore = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < tree.children.size(); i++) {
            double visits = tree.children.get(i).visit_count;
            if (visits <= 0) continue;
            PlayerAction action = tree.actions.get(i);
            if (!actionRespectsCurrentStance(action)) continue;

            double avgEval = tree.children.get(i).accum_evaluation / Math.max(1.0, visits);
            int pref = preferenceScore(action);
            if (!preferredActions.isEmpty() && pref < 0) continue;

            double score = (visits * 100.0) + (pref * 18.0) + (avgEval * 90.0);
            if (score > bestScore) {
                bestScore = score;
                bestIdx = i;
            }
        }
        if (bestIdx == -1) return super.getMostVisitedActionIdx();
        return bestIdx;
    }

    /**
     * Scores an action according to current stance intent and production priorities.
     */
    private int preferenceScore(PlayerAction pa) {
        if (pa == null || preferredActions.isEmpty()) return 0;
        int score = 0;
        boolean rangedAttackHeavyDefense = isRangedAttackHeavyDefenseModeActive();
        int myHeavy = countUnits(activePlayer, gs_to_start_from, heavyType);
        int myRanged = countUnits(activePlayer, gs_to_start_from, rangedType);

        for (Pair<Unit, UnitAction> uaa : pa.getActions()) {
            Unit u = uaa.m_a;
            UnitAction a = uaa.m_b;
            int type = a == null ? -1 : a.getType();
            Intent intent = classifyIntent(u, a, u.getPlayer());

            score += scoreEconomyActionPreference(type);
            score += scoreProductionActionPreference(a, myHeavy, myRanged, rangedAttackHeavyDefense);
            score += scoreBaseProximityPreference(u, a, type);
            score += scoreDefenseIntentPreference(intent);
            score += scoreSplitFormationPreference(u, intent, rangedAttackHeavyDefense, myHeavy);
            score += scoreStanceIntentPreference(intent);
        }
        return score;
    }

    private int scoreEconomyActionPreference(int actionType) {
        int score = 0;
        if (preferredActions.contains("HARVEST") && actionType == UnitAction.TYPE_HARVEST) score += 2;
        if (preferredActions.contains("RETURN") && actionType == UnitAction.TYPE_RETURN) score += 2;
        return score;
    }

    private int scoreProductionActionPreference(UnitAction action, int myHeavy, int myRanged,
                                                boolean rangedAttackHeavyDefense) {
        if (action == null || action.getType() != UnitAction.TYPE_PRODUCE || action.getUnitType() == null) return 0;

        int score = 0;
        String produced = action.getUnitType().name.toUpperCase();
        if (preferredActions.contains("PRODUCE_HEAVY") && "HEAVY".equals(produced)) score += 5;
        if (preferredActions.contains("PRODUCE_RANGED") && "RANGED".equals(produced)) score += 5;
        if (preferredActions.contains("PRODUCE_WORKER") && "WORKER".equals(produced)) score += 4;
        if (preferredActions.contains("BUILD_BARRACKS") && "BARRACKS".equals(produced)) score += 6;
        if (preferredActions.contains("PRODUCE_" + preferredUnit) && produced.equals(preferredUnit)) score += 5;

        if (currentStance == Stance.ATTACK) {
            if ("RANGED".equals(produced)) score += myRanged <= myHeavy ? 4 : 2;
            if ("HEAVY".equals(produced)) score += myHeavy < myRanged ? 4 : 2;
        }

        if (rangedAttackHeavyDefense) {
            if ("RANGED".equals(produced)) score += 8;
            if ("HEAVY".equals(produced)) {
                score += myHeavy < HEAVY_DEFENSE_TARGET ? 4 : 0;
                if (myHeavy >= HEAVY_DEFENSE_TARGET + 1) score -= 3;
            }
        }
        return score;
    }

    private int scoreBaseProximityPreference(Unit unit, UnitAction action, int actionType) {
        if (unit == null || action == null) return 0;
        if (!preferredActions.contains("ATTACK_NEAR_BASE")) return 0;
        if (actionType != UnitAction.TYPE_ATTACK_LOCATION) return 0;
        if (!isActionNearAnyOwnBase(action, unit.getPlayer(), baseDefenseRadius())) return 0;
        return 3;
    }

    private int scoreDefenseIntentPreference(Intent intent) {
        if (preferredActions.contains("DEFEND_BASE") && intent == Intent.DEFENSE) return 2;
        return 0;
    }

    private int scoreSplitFormationPreference(Unit unit, Intent intent,
                                              boolean rangedAttackHeavyDefense, int myHeavy) {
        if (!rangedAttackHeavyDefense || unit == null || unit.getType() == null) return 0;
        if (unit.getType() == rangedType) {
            if (intent == Intent.OFFENSE) return 10;
            if (intent == Intent.DEFENSE) return -12;
            return 0;
        }
        if (unit.getType() == heavyType) {
            if (intent == Intent.DEFENSE) return 8;
            if (intent == Intent.OFFENSE && myHeavy <= HEAVY_DEFENSE_TARGET) return -8;
            if (intent == Intent.OFFENSE && myHeavy > HEAVY_DEFENSE_TARGET) return -3;
        }
        return 0;
    }

    private int scoreStanceIntentPreference(Intent intent) {
        if (currentStance == Stance.DEFEND) {
            if (intent == Intent.DEFENSE) return 3;
            if (intent == Intent.OFFENSE) return -10;
            return 0;
        }
        if (intent == Intent.OFFENSE) return 5;
        if (intent == Intent.DEFENSE) return -10;
        return 0;
    }

    /**
     * Filters out action sets that violate the current global stance.
     */
    private boolean actionRespectsCurrentStance(PlayerAction pa) {
        if (pa == null) return false;

        int offensive = 0;
        int defensive = 0;

        for (Pair<Unit, UnitAction> uaa : pa.getActions()) {
            Unit unit = uaa.m_a;
            if (!unit.getType().canMove) continue;
            Intent intent = classifyIntent(unit, uaa.m_b, unit.getPlayer());
            if (intent == Intent.OFFENSE) offensive++;
            if (intent == Intent.DEFENSE) defensive++;
        }

        if (currentStance == Stance.DEFEND) return offensive == 0;

        int combatUnits = countMyCombatUnits(activePlayer, gs_to_start_from);
        if (combatUnits <= 0) return true;
        if (offensive > 0) return true;
        if (producesCombatUnit(pa)) return true;
        return defensive == 0;
    }

    private boolean producesCombatUnit(PlayerAction pa) {
        if (pa == null) return false;
        for (Pair<Unit, UnitAction> uaa : pa.getActions()) {
            UnitAction action = uaa.m_b;
            if (action == null || action.getType() != UnitAction.TYPE_PRODUCE || action.getUnitType() == null) {
                continue;
            }
            UnitType produced = action.getUnitType();
            if (produced.canAttack && !produced.canHarvest) return true;
        }
        return false;
    }

    /**
     * Maps low-level unit actions into strategic intent buckets.
     */
    private Intent classifyIntent(Unit unit, UnitAction action, int player) {
        if (unit == null || action == null) return Intent.NEUTRAL;
        int t = action.getType();

        if (t == UnitAction.TYPE_HARVEST || t == UnitAction.TYPE_RETURN) return Intent.ECONOMY;
        if (t == UnitAction.TYPE_PRODUCE) {
            if (action.getUnitType() != null) {
                UnitType produced = action.getUnitType();
                if (produced == workerType || produced == barracksType || produced == baseType) {
                    return Intent.ECONOMY;
                }
            }
            return currentStance == Stance.ATTACK ? Intent.OFFENSE : Intent.DEFENSE;
        }
        if (t == UnitAction.TYPE_ATTACK_LOCATION) {
            return isActionNearAnyOwnBase(action, player, baseDefenseRadius())
                    ? Intent.DEFENSE
                    : Intent.OFFENSE;
        }
        if (t == UnitAction.TYPE_MOVE) {
            boolean towardEnemy = moveReducesDistanceToEnemyBase(unit, action, player);
            boolean towardOwnBase = moveReducesDistanceToOwnBase(unit, action, player);
            if (towardEnemy && !towardOwnBase) return Intent.OFFENSE;
            if (towardOwnBase && !towardEnemy) return Intent.DEFENSE;
        }
        return Intent.NEUTRAL;
    }

    private boolean moveReducesDistanceToEnemyBase(Unit unit, UnitAction a, int player) {
        int before = minDistanceToEnemyBase(unit.getX(), unit.getY(), player);
        int after = minDistanceToEnemyBase(projectedX(unit, a), projectedY(unit, a), player);
        return after < before;
    }

    private boolean moveReducesDistanceToOwnBase(Unit unit, UnitAction a, int player) {
        if (gs_to_start_from == null) return false;
        int nx = projectedX(unit, a);
        int ny = projectedY(unit, a);

        int before = Integer.MAX_VALUE;
        int after = Integer.MAX_VALUE;
        for (Unit u : gs_to_start_from.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && u.getType() == baseType) {
                before = Math.min(before, Math.abs(unit.getX() - u.getX()) + Math.abs(unit.getY() - u.getY()));
                after = Math.min(after, Math.abs(nx - u.getX()) + Math.abs(ny - u.getY()));
            }
        }
        return after < before;
    }

    private boolean isActionNearAnyOwnBase(UnitAction a, int player, int radius) {
        if (gs_to_start_from == null) return false;
        for (Unit u : gs_to_start_from.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && u.getType() == baseType) {
                int d = Math.abs(u.getX() - a.getLocationX()) + Math.abs(u.getY() - a.getLocationY());
                if (d <= radius) return true;
            }
        }
        return false;
    }

    private int projectedX(Unit u, UnitAction a) {
        if (a == null || a.getType() != UnitAction.TYPE_MOVE) return u.getX();
        int dir = a.getDirection();
        if (dir >= 0 && dir < 4) return u.getX() + UnitAction.DIRECTION_OFFSET_X[dir];
        return u.getX();
    }

    private int projectedY(Unit u, UnitAction a) {
        if (a == null || a.getType() != UnitAction.TYPE_MOVE) return u.getY();
        int dir = a.getDirection();
        if (dir >= 0 && dir < 4) return u.getY() + UnitAction.DIRECTION_OFFSET_Y[dir];
        return u.getY();
    }

    /**
     * True when the current map is 8x8 (used for the tiny-map worker-rush policy).
     */
    private boolean isEightByEightMap(GameState gs) {
        if (gs == null || gs.getPhysicalGameState() == null) return false;
        return gs.getPhysicalGameState().getWidth() == 8 && gs.getPhysicalGameState().getHeight() == 8;
    }

    private int distanceToClosest(Unit from, List<Unit> targets) {
        if (targets == null || targets.isEmpty()) return Integer.MAX_VALUE;
        int best = Integer.MAX_VALUE;
        for (Unit t : targets) {
            int d = Math.abs(from.getX() - t.getX()) + Math.abs(from.getY() - t.getY());
            if (d < best) best = d;
        }
        return best;
    }

    private boolean isGettingRushed(Snapshot snapshot) {
        int rushAlertRadius = rushAlertRadius(snapshot.gs);
        int enemyThreat = 0;
        for (Unit enemy : snapshot.gs.getPhysicalGameState().getUnits()) {
            if (enemy.getPlayer() < 0 || enemy.getPlayer() == snapshot.player) continue;
            boolean isThreat = enemy.getType().canAttack || enemy.getType() == workerType;
            if (!isThreat) continue;
            int d = distanceToClosest(enemy, snapshot.myBases);
            if (d <= rushAlertRadius) enemyThreat++;
        }
        return enemyThreat >= 2 || (enemyThreat > 0 && enemyThreat >= snapshot.myCombat);
    }

    private int countMyCombatUnits(int player, GameState gs) {
        if (gs == null) return 0;
        int n = 0;
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && isCombatType(u.getType())) {
                n++;
            }
        }
        return n;
    }

    private boolean isCombatType(UnitType type) {
        return type != null && type.canAttack && !type.canHarvest;
    }

    private int countUnits(int player, GameState gs, UnitType type) {
        if (gs == null || type == null) return 0;
        int n = 0;
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() == player && u.getType() == type) n++;
        }
        return n;
    }

    private boolean canAffordUnitTypeNow(int player, UnitType type, PlayerAction pending, GameState gs) {
        if (type == null || gs == null || player < 0) return false;
        ResourceUsage reserved = gs.getResourceUsage();
        int alreadyReserved = reserved == null ? 0 : reserved.getResourcesUsed(player);
        alreadyReserved = Math.max(alreadyReserved, committedResourcesFromAssignments(player, gs));
        int pendingReserved = (pending == null || pending.getResourceUsage() == null)
                ? 0
                : pending.getResourceUsage().getResourcesUsed(player);
        int available = gs.getPlayer(player).getResources() - alreadyReserved - pendingReserved;
        return available >= type.cost;
    }

    private int committedResourcesFromAssignments(int player, GameState gs) {
        if (gs == null || player < 0) return 0;
        int committed = 0;
        for (UnitActionAssignment uaa : gs.getUnitActions().values()) {
            if (uaa == null || uaa.unit == null || uaa.action == null) continue;
            if (uaa.unit.getPlayer() != player) continue;
            if (uaa.action.getType() == UnitAction.TYPE_PRODUCE && uaa.action.getUnitType() != null) {
                committed += uaa.action.getUnitType().cost;
            }
        }
        return committed;
    }

    /**
     * Opening completion condition.
     */
    private boolean openingGoalsMet(Snapshot snapshot) {
        return snapshot.myWorkers >= openingWorkerTarget(snapshot.gs)
                && snapshot.myBarracks >= 1
                && snapshot.myRanged >= openingRangedTarget(snapshot.gs)
                && snapshot.myHeavy >= openingHeavyTarget(snapshot.gs);
    }

    private int minDistanceToEnemyBase(int x, int y, int player) {
        if (gs_to_start_from == null) return 9999;
        int best = 9999;
        for (Unit u : gs_to_start_from.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() >= 0 && u.getPlayer() != player && u.getType() == baseType) {
                int d = Math.abs(x - u.getX()) + Math.abs(y - u.getY());
                best = Math.min(best, d);
            }
        }
        return best;
    }

    private int minMyCombatDistanceToEnemyBase(int player, GameState gs, List<Unit> enemyBases) {
        if (enemyBases == null || enemyBases.isEmpty()) return Integer.MAX_VALUE;
        int best = Integer.MAX_VALUE;
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() != player || !u.getType().canAttack || u.getType().canHarvest) continue;
            best = Math.min(best, distanceToClosest(u, enemyBases));
        }
        return best;
    }

    private int minEnemyCombatDistanceToMyBase(int player, GameState gs, List<Unit> myBases) {
        if (myBases == null || myBases.isEmpty()) return Integer.MAX_VALUE;
        int best = Integer.MAX_VALUE;
        for (Unit u : gs.getPhysicalGameState().getUnits()) {
            if (u.getPlayer() < 0 || u.getPlayer() == player || !u.getType().canAttack || u.getType().canHarvest) continue;
            best = Math.min(best, distanceToClosest(u, myBases));
        }
        return best;
    }

    private Map<Long, Pair<Unit, UnitAction>> toActionMap(PlayerAction pa) {
        Map<Long, Pair<Unit, UnitAction>> map = new HashMap<>();
        if (pa == null) return map;
        for (Pair<Unit, UnitAction> uaa : pa.getActions()) {
            map.put(uaa.m_a.getID(), uaa);
        }
        return map;
    }

    private boolean addIfConsistent(PlayerAction out, Unit u, UnitAction a, GameState gs) {
        if (u == null || a == null) return false;
        ResourceUsage ru = a.resourceUsage(u, gs.getPhysicalGameState());
        ResourceUsage reserved = gs.getResourceUsage().mergeIntoNew(out.getResourceUsage());
        if (reserved.consistentWith(ru, gs)) {
            out.addUnitAction(u, a);
            out.getResourceUsage().merge(ru);
            return true;
        }
        return false;
    }

    private static int getEnvInt(String key, int defaultValue) {
        String raw = System.getenv(key);
        if (raw == null || raw.trim().isEmpty()) return defaultValue;
        try {
            return Integer.parseInt(raw.trim());
        } catch (NumberFormatException ignored) {
            return defaultValue;
        }
    }

    private int openingEndTick(int mapArea) {
        if (mapArea <= 64) return 260;
        if (mapArea <= 256) return 300;
        return 460;
    }

    private int openingWorkersBeforeBarracks(GameState gs) {
        int area = mapArea(gs);
        if (area <= 64) return 0;
        return OPENING_WORKERS_BEFORE_BARRACKS_DEFAULT;
    }

    private int openingWorkerTarget(GameState gs) {
        int area = mapArea(gs);
        if (area <= 256) return 3;
        return 5;
    }

    private int openingRangedTarget(GameState gs) {
        int area = mapArea(gs);
        if (area <= 144) return OPENING_RANGED_TARGET_DEFAULT;
        return 2;
    }

    private int openingHeavyTarget(GameState gs) {
        return OPENING_HEAVY_TARGET_DEFAULT;
    }

    private int rushAlertRadius(GameState gs) {
        if (gs == null || gs.getPhysicalGameState() == null) return RUSH_ALERT_RADIUS_DEFAULT;
        int minDim = Math.min(gs.getPhysicalGameState().getWidth(), gs.getPhysicalGameState().getHeight());
        int scaled = (int) Math.round(minDim * 0.75);
        return Math.max(5, Math.min(12, scaled));
    }

    private int baseDefenseRadius() {
        if (gs_to_start_from == null || gs_to_start_from.getPhysicalGameState() == null) {
            return BASE_DEFENSE_RADIUS_DEFAULT;
        }
        int minDim = Math.min(gs_to_start_from.getPhysicalGameState().getWidth(),
                gs_to_start_from.getPhysicalGameState().getHeight());
        if (minDim <= 8) return 3;
        if (minDim <= 16) return BASE_DEFENSE_RADIUS_DEFAULT;
        return 5;
    }

    private int mapArea(GameState gs) {
        if (gs == null || gs.getPhysicalGameState() == null) return 0;
        return gs.getPhysicalGameState().getWidth() * gs.getPhysicalGameState().getHeight();
    }

    private boolean isRangedAttackHeavyDefenseModeActive() {
        return llmRangedAttackHeavyDefenseMode
                && currentStance == Stance.ATTACK
                && gs_to_start_from != null
                && mapArea(gs_to_start_from) >= LARGE_MAP_AREA_THRESHOLD;
    }

    private void recordLlmSuccessLocked() {
        llmFailureStreak = 0;
        llmCooldownUntilTick = -1;
    }

    private void recordLlmFailureLocked(int tick) {
        llmFailureStreak = Math.min(llmFailureStreak + 1, 8);
        int multiplier = Math.min(4, llmFailureStreak);
        int cooldown = LLM_FAILURE_COOLDOWN_BASE_TICKS * multiplier;
        cooldown = Math.max(10, Math.min(LLM_FAILURE_COOLDOWN_MAX_TICKS, cooldown));
        llmCooldownUntilTick = Math.max(llmCooldownUntilTick, tick + cooldown);
    }

    private boolean isLlmPreferredUnitActive() {
        return llmPreferredUnitOverride != null
                && !llmPreferredUnitOverride.isEmpty();
    }

    @Override
    public void reset() {
        super.reset();
        lastConsultTick = -9999;
        activePlayer = 0;
        openingComplete = false;
        finishMode = false;
        currentStance = Stance.DEFEND;
        preferredUnit = "RANGED";
        preferredReason = "Defensive opening";
        preferredActions.clear();
        preferredActions.add("PRODUCE_HEAVY");
        preferredActions.add("PRODUCE_RANGED");
        preferredActions.add("DEFEND_BASE");
        llmPreferredUnitOverride = null;
        llmRangedAttackHeavyDefenseMode = false;
        synchronized (this) {
            llmRequestGeneration++;
            llmRequestInFlight = false;
            pendingLlmResponse = null;
            pendingLlmResponseTick = -1;
            llmFailureStreak = 0;
            llmCooldownUntilTick = -1;
        }
    }

    /**
     * Clones runtime strategy state so search rollouts and future decisions stay behaviorally consistent.
     */
    @Override
    public AI clone() {
        MCTSAgent cloned = new MCTSAgent(utt);
        cloned.setTimeBudget(TIME_BUDGET);
        cloned.setIterationsBudget(ITERATIONS_BUDGET);
        cloned.MAXSIMULATIONTIME = MAXSIMULATIONTIME;
        cloned.MAX_TREE_DEPTH = MAX_TREE_DEPTH;
        cloned.epsilon_l = epsilon_l;
        cloned.epsilon_g = epsilon_g;
        cloned.epsilon_0 = epsilon_0;
        cloned.initial_epsilon_l = initial_epsilon_l;
        cloned.initial_epsilon_g = initial_epsilon_g;
        cloned.initial_epsilon_0 = initial_epsilon_0;
        cloned.currentStance = currentStance;
        cloned.preferredUnit = preferredUnit;
        cloned.preferredReason = preferredReason;
        cloned.preferredActions = new HashSet<>(preferredActions);
        cloned.activePlayer = activePlayer;
        cloned.openingComplete = openingComplete;
        cloned.finishMode = finishMode;
        cloned.resolvedOllamaModel = resolvedOllamaModel;
        cloned.llmPreferredUnitOverride = llmPreferredUnitOverride;
        cloned.llmRangedAttackHeavyDefenseMode = llmRangedAttackHeavyDefenseMode;
        cloned.lastConsultTick = lastConsultTick;
        cloned.llmFailureStreak = llmFailureStreak;
        cloned.llmCooldownUntilTick = llmCooldownUntilTick;
        cloned.llmRequestGeneration = llmRequestGeneration;
        cloned.applyStanceBiases();
        return cloned;
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}
