package ai.abstraction.submissions.chase;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import ai.abstraction.HeavyRush;
import ai.abstraction.LightRush;
import ai.abstraction.RangedRush;
import ai.abstraction.WorkerRush;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.core.AIWithComputationBudget;
import ai.core.ParameterSpecification;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.units.Unit;
import rts.units.UnitTypeTable;

public class ChaseBotPacked extends AIWithComputationBudget {

    private final UnitTypeTable utt;
    private final StrategyAdvisor advisor;
    private final ChaseBotConfig config;
    private final DeterministicStrategyEngine strategyEngine;
    private final AStarPathFinding pathFinding;

    private final WorkerRush workerRush;
    private final LightRush lightRush;
    private final HeavyRush heavyRush;
    private final RangedRush rangedRush;

    private int lastConsultationTick = -9999;
    private String lastPhase = "opening";
    private AdvisorRecommendation lastAppliedRecommendation = AdvisorRecommendation.neutral();
    private MacroDecision lastDecision = null;

    public ChaseBotPacked(UnitTypeTable utt) {
        this(utt, ChaseBotConfig.fromEnvironment());
    }

    private ChaseBotPacked(UnitTypeTable utt, ChaseBotConfig config) {
        this(utt, defaultAdvisor(config), config);
    }

    public ChaseBotPacked(UnitTypeTable utt, StrategyAdvisor advisor, ChaseBotConfig config) {
        super(100, -1);
        this.utt = utt;
        this.config = config;
        this.advisor = advisor != null ? advisor : new NoOpAdvisor();
        this.strategyEngine = new DeterministicStrategyEngine();
        this.pathFinding = new AStarPathFinding();
        this.workerRush = new WorkerRush(utt, pathFinding);
        this.lightRush = new LightRush(utt, pathFinding);
        this.heavyRush = new HeavyRush(utt, pathFinding);
        this.rangedRush = new RangedRush(utt, pathFinding);
    }

    private static StrategyAdvisor defaultAdvisor(ChaseBotConfig config) {
        if (config.isAdvisorEnabled()) {
            return new OllamaAdvisor(config);
        }
        return new NoOpAdvisor();
    }

    @Override
    public void reset() {
        lastConsultationTick = -9999;
        lastPhase = "opening";
        lastAppliedRecommendation = AdvisorRecommendation.neutral();
        lastDecision = null;
        workerRush.reset();
        lightRush.reset();
        heavyRush.reset();
        rangedRush.reset();
    }

    @Override
    public AI clone() {
        ChaseBotPacked clone = new ChaseBotPacked(utt, advisor.copy(), config);
        clone.setTimeBudget(TIME_BUDGET);
        clone.setIterationsBudget(ITERATIONS_BUDGET);
        return clone;
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();
        parameters.add(new ParameterSpecification("TimeBudget", int.class, 100));
        parameters.add(new ParameterSpecification("IterationsBudget", int.class, -1));
        return parameters;
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) throws Exception {
        if (!gs.canExecuteAnyAction(player)) {
            return new PlayerAction();
        }

        ChaseGameSnapshot snapshot = ChaseGameSnapshot.fromGameState(player, gs, pathFinding);
        maybeRefreshRecommendation(snapshot);
        lastDecision = strategyEngine.decide(snapshot, lastAppliedRecommendation);

        AI delegate = selectDelegate(lastDecision, snapshot);
        PlayerAction action = delegate.getAction(player, gs);
        return action != null ? action : new PlayerAction();
    }

    public AdvisorRecommendation getLastAppliedRecommendation() {
        return lastAppliedRecommendation;
    }

    public MacroDecision getLastDecision() {
        return lastDecision;
    }

    private void maybeRefreshRecommendation(ChaseGameSnapshot snapshot) {
        String phase = determinePhase(snapshot);
        boolean phaseChanged = !phase.equals(lastPhase);
        boolean intervalElapsed =
                snapshot.getTime() - lastConsultationTick >= config.getConsultationInterval();

        if (!phaseChanged && !intervalElapsed) {
            return;
        }

        lastPhase = phase;
        lastConsultationTick = snapshot.getTime();
        try {
            lastAppliedRecommendation = AdvisorRecommendation.sanitize(advisor.advise(snapshot));
        } catch (Exception ex) {
            lastAppliedRecommendation = AdvisorRecommendation.neutral();
        }
    }

    private String determinePhase(ChaseGameSnapshot snapshot) {
        if (snapshot.getNearestEnemyToBase() <= 5) {
            return "defense";
        }
        if (snapshot.getTime() < 250 && snapshot.getMyBarracks() == 0) {
            return "opening";
        }
        if (snapshot.getTime() > 1200 || snapshot.getMyBarracks() >= 2) {
            return "late";
        }
        return "midgame";
    }

    private AI selectDelegate(MacroDecision decision, ChaseGameSnapshot snapshot) {
        if (decision == null) {
            return lightRush;
        }
        if (snapshot.getMapMaxDimension() <= 8) {
            if (snapshot.getTime() < config.getCoacOpeningTicks()) {
                return workerRush;
            }
            return lightRush;
        }
        if (snapshot.getMapMaxDimension() <= 16) {
            if (shouldUseSmallMapPressureDelegate(snapshot)) {
                return rangedRush;
            }
            return workerRush;
        }
        switch (decision.getStrategy()) {
            case WORKER_RUSH:
                return workerRush;
            case HEAVY_RUSH:
                return heavyRush;
            case RANGED_RUSH:
                return rangedRush;
            case LIGHT_RUSH:
            default:
                return lightRush;
        }
    }

    private boolean shouldUseSmallMapPressureDelegate(ChaseGameSnapshot snapshot) {
        if (snapshot.getMapMaxDimension() > 8) {
            return false;
        }
        if (snapshot.getTime() < 220) {
            return false;
        }
        if (snapshot.getEnemyBarracks() == 0) {
            return false;
        }
        if (!snapshot.isPathToEnemyOpen()) {
            return false;
        }
        return snapshot.getEnemyLights() + snapshot.getEnemyHeavies() + snapshot.getEnemyRanged() > 0;
    }
}

final class AdvisorRecommendation {

    private static final AdvisorRecommendation NEUTRAL =
            new AdvisorRecommendation(0.0, 0.0, null, null);

    private final double attackBias;
    private final double economyBias;
    private final MacroStrategy preferredStrategy;
    private final UnitPreference unitPreference;

    private AdvisorRecommendation(
            double attackBias,
            double economyBias,
            MacroStrategy preferredStrategy,
            UnitPreference unitPreference) {
        this.attackBias = attackBias;
        this.economyBias = economyBias;
        this.preferredStrategy = preferredStrategy;
        this.unitPreference = unitPreference;
    }

    public static AdvisorRecommendation neutral() {
        return NEUTRAL;
    }

    public static AdvisorRecommendation of(
            double attackBias,
            double economyBias,
            MacroStrategy preferredStrategy,
            UnitPreference unitPreference) {
        return sanitize(new AdvisorRecommendation(attackBias, economyBias, preferredStrategy, unitPreference));
    }

    public static AdvisorRecommendation sanitize(AdvisorRecommendation recommendation) {
        if (recommendation == null) {
            return neutral();
        }
        double safeAttack = clampBias(recommendation.attackBias);
        double safeEconomy = clampBias(recommendation.economyBias);
        return new AdvisorRecommendation(
                safeAttack,
                safeEconomy,
                recommendation.preferredStrategy,
                recommendation.unitPreference);
    }

    private static double clampBias(double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            return 0.0;
        }
        return Math.max(-3.0, Math.min(3.0, value));
    }

    public boolean isNeutral() {
        return attackBias == 0.0
                && economyBias == 0.0
                && preferredStrategy == null
                && unitPreference == null;
    }

    public double getAttackBias() {
        return attackBias;
    }

    public double getEconomyBias() {
        return economyBias;
    }

    public MacroStrategy getPreferredStrategy() {
        return preferredStrategy;
    }

    public UnitPreference getUnitPreference() {
        return unitPreference;
    }
}

enum BotPosture {
    DEFENSIVE,
    BALANCED,
    AGGRESSIVE
}

final class ChaseBotConfig {

    private final boolean advisorEnabled;
    private final String ollamaModel;
    private final int consultationInterval;
    private final int connectTimeoutMs;
    private final int readTimeoutMs;
    private final int advisorCacheEntries;
    private final int coacOpeningTicks;

    private ChaseBotConfig(Builder builder) {
        this.advisorEnabled = builder.advisorEnabled;
        this.ollamaModel = builder.ollamaModel;
        this.consultationInterval = builder.consultationInterval;
        this.connectTimeoutMs = builder.connectTimeoutMs;
        this.readTimeoutMs = builder.readTimeoutMs;
        this.advisorCacheEntries = builder.advisorCacheEntries;
        this.coacOpeningTicks = builder.coacOpeningTicks;
    }

    public static ChaseBotConfig fromEnvironment() {
        Builder builder = builder();
        builder.setAdvisorEnabled(Boolean.parseBoolean(
                System.getenv().getOrDefault("CHASEBOT_ADVISOR_ENABLED", "true")));
        builder.setOllamaModel(System.getenv().getOrDefault(
                "CHASEBOT_OLLAMA_MODEL",
                System.getenv().getOrDefault("OLLAMA_MODEL", "llama3.1:8b")));
        builder.setConsultationInterval(parseInt("CHASEBOT_CONSULT_INTERVAL", 250));
        builder.setConnectTimeoutMs(parseInt("CHASEBOT_CONNECT_TIMEOUT_MS", 1500));
        builder.setReadTimeoutMs(parseInt("CHASEBOT_READ_TIMEOUT_MS", 5000));
        builder.setAdvisorCacheEntries(parseInt("CHASEBOT_ADVISOR_CACHE", 128));
        builder.setCoacOpeningTicks(parseInt("CHASEBOT_COAC_OPENING_TICKS", 210));
        return builder.build();
    }

    private static int parseInt(String envName, int fallback) {
        try {
            return Integer.parseInt(System.getenv().getOrDefault(envName, String.valueOf(fallback)));
        } catch (NumberFormatException ex) {
            return fallback;
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    public boolean isAdvisorEnabled() {
        return advisorEnabled;
    }

    public String getOllamaModel() {
        return ollamaModel;
    }

    public int getConsultationInterval() {
        return consultationInterval;
    }

    public int getConnectTimeoutMs() {
        return connectTimeoutMs;
    }

    public int getReadTimeoutMs() {
        return readTimeoutMs;
    }

    public int getAdvisorCacheEntries() {
        return advisorCacheEntries;
    }

    public int getCoacOpeningTicks() {
        return coacOpeningTicks;
    }

    static final class Builder {
        private boolean advisorEnabled = true;
        private String ollamaModel = "llama3.1:8b";
        private int consultationInterval = 250;
        private int connectTimeoutMs = 1500;
        private int readTimeoutMs = 5000;
        private int advisorCacheEntries = 128;
        private int coacOpeningTicks = 210;

        Builder setAdvisorEnabled(boolean advisorEnabled) {
            this.advisorEnabled = advisorEnabled;
            return this;
        }

        Builder setOllamaModel(String ollamaModel) {
            this.ollamaModel = ollamaModel;
            return this;
        }

        Builder setConsultationInterval(int consultationInterval) {
            this.consultationInterval = consultationInterval;
            return this;
        }

        Builder setConnectTimeoutMs(int connectTimeoutMs) {
            this.connectTimeoutMs = connectTimeoutMs;
            return this;
        }

        Builder setReadTimeoutMs(int readTimeoutMs) {
            this.readTimeoutMs = readTimeoutMs;
            return this;
        }

        Builder setAdvisorCacheEntries(int advisorCacheEntries) {
            this.advisorCacheEntries = advisorCacheEntries;
            return this;
        }

        Builder setCoacOpeningTicks(int coacOpeningTicks) {
            this.coacOpeningTicks = Math.max(0, coacOpeningTicks);
            return this;
        }

        ChaseBotConfig build() {
            return new ChaseBotConfig(this);
        }
    }
}

final class ChaseGameSnapshot {

    private final int mapWidth;
    private final int mapHeight;
    private final int time;
    private final int myResources;
    private final int enemyResources;
    private final int myWorkers;
    private final int myLights;
    private final int myHeavies;
    private final int myRanged;
    private final int myBases;
    private final int myBarracks;
    private final int enemyWorkers;
    private final int enemyLights;
    private final int enemyHeavies;
    private final int enemyRanged;
    private final int enemyBases;
    private final int enemyBarracks;
    private final boolean pathToEnemyOpen;
    private final int nearestEnemyToBase;
    private final int nearbyResources;

    private ChaseGameSnapshot(Builder builder) {
        this.mapWidth = builder.mapWidth;
        this.mapHeight = builder.mapHeight;
        this.time = builder.time;
        this.myResources = builder.myResources;
        this.enemyResources = builder.enemyResources;
        this.myWorkers = builder.myWorkers;
        this.myLights = builder.myLights;
        this.myHeavies = builder.myHeavies;
        this.myRanged = builder.myRanged;
        this.myBases = builder.myBases;
        this.myBarracks = builder.myBarracks;
        this.enemyWorkers = builder.enemyWorkers;
        this.enemyLights = builder.enemyLights;
        this.enemyHeavies = builder.enemyHeavies;
        this.enemyRanged = builder.enemyRanged;
        this.enemyBases = builder.enemyBases;
        this.enemyBarracks = builder.enemyBarracks;
        this.pathToEnemyOpen = builder.pathToEnemyOpen;
        this.nearestEnemyToBase = builder.nearestEnemyToBase;
        this.nearbyResources = builder.nearbyResources;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static ChaseGameSnapshot fromGameState(int player, GameState gs, AStarPathFinding pathFinding) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player me = gs.getPlayer(player);
        Player enemy = gs.getPlayer(1 - player);

        Builder builder = builder()
                .setMapWidth(pgs.getWidth())
                .setMapHeight(pgs.getHeight())
                .setTime(gs.getTime())
                .setMyResources(me.getResources())
                .setEnemyResources(enemy != null ? enemy.getResources() : 0);

        Unit myBase = null;
        Unit enemyBase = null;

        for (Unit unit : pgs.getUnits()) {
            String type = unit.getType().name;
            if (unit.getPlayer() == player) {
                if ("Worker".equals(type)) {
                    builder.setMyWorkers(builder.myWorkers + 1);
                } else if ("Light".equals(type)) {
                    builder.setMyLights(builder.myLights + 1);
                } else if ("Heavy".equals(type)) {
                    builder.setMyHeavies(builder.myHeavies + 1);
                } else if ("Ranged".equals(type)) {
                    builder.setMyRanged(builder.myRanged + 1);
                } else if ("Base".equals(type)) {
                    builder.setMyBases(builder.myBases + 1);
                    if (myBase == null) {
                        myBase = unit;
                    }
                } else if ("Barracks".equals(type)) {
                    builder.setMyBarracks(builder.myBarracks + 1);
                }
            } else if (unit.getPlayer() >= 0) {
                if ("Worker".equals(type)) {
                    builder.setEnemyWorkers(builder.enemyWorkers + 1);
                } else if ("Light".equals(type)) {
                    builder.setEnemyLights(builder.enemyLights + 1);
                } else if ("Heavy".equals(type)) {
                    builder.setEnemyHeavies(builder.enemyHeavies + 1);
                } else if ("Ranged".equals(type)) {
                    builder.setEnemyRanged(builder.enemyRanged + 1);
                } else if ("Base".equals(type)) {
                    builder.setEnemyBases(builder.enemyBases + 1);
                    if (enemyBase == null) {
                        enemyBase = unit;
                    }
                } else if ("Barracks".equals(type)) {
                    builder.setEnemyBarracks(builder.enemyBarracks + 1);
                }
            }
        }

        int nearbyResourceCount = 0;
        int closestEnemyToBase = Integer.MAX_VALUE;
        if (myBase != null) {
            for (Unit unit : pgs.getUnits()) {
                if (unit.getType().isResource) {
                    int distance = manhattan(myBase, unit);
                    if (distance <= 8) {
                        nearbyResourceCount++;
                    }
                } else if (unit.getPlayer() >= 0 && unit.getPlayer() != player) {
                    closestEnemyToBase = Math.min(closestEnemyToBase, manhattan(myBase, unit));
                }
            }
        }

        builder.setNearbyResources(nearbyResourceCount);
        builder.setNearestEnemyToBase(closestEnemyToBase == Integer.MAX_VALUE ? 9999 : closestEnemyToBase);

        boolean pathOpen = true;
        if (myBase != null && enemyBase != null) {
            int distance = pathFinding.findDistToPositionInRange(
                    myBase,
                    enemyBase.getPosition(pgs),
                    1,
                    gs,
                    gs.getResourceUsage());
            pathOpen = distance >= 0;
        }
        builder.setPathToEnemyOpen(pathOpen);

        return builder.build();
    }

    private static int manhattan(Unit a, Unit b) {
        return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
    }

    public int getMapWidth() {
        return mapWidth;
    }

    public int getMapHeight() {
        return mapHeight;
    }

    public int getMapMaxDimension() {
        return Math.max(mapWidth, mapHeight);
    }

    public int getTime() {
        return time;
    }

    public int getMyResources() {
        return myResources;
    }

    public int getEnemyResources() {
        return enemyResources;
    }

    public int getMyWorkers() {
        return myWorkers;
    }

    public int getMyLights() {
        return myLights;
    }

    public int getMyHeavies() {
        return myHeavies;
    }

    public int getMyRanged() {
        return myRanged;
    }

    public int getMyBases() {
        return myBases;
    }

    public int getMyBarracks() {
        return myBarracks;
    }

    public int getEnemyWorkers() {
        return enemyWorkers;
    }

    public int getEnemyLights() {
        return enemyLights;
    }

    public int getEnemyHeavies() {
        return enemyHeavies;
    }

    public int getEnemyRanged() {
        return enemyRanged;
    }

    public int getEnemyBases() {
        return enemyBases;
    }

    public int getEnemyBarracks() {
        return enemyBarracks;
    }

    public boolean isPathToEnemyOpen() {
        return pathToEnemyOpen;
    }

    public int getNearestEnemyToBase() {
        return nearestEnemyToBase;
    }

    public int getNearbyResources() {
        return nearbyResources;
    }

    public int getMyMilitaryStrength() {
        return myWorkers + (2 * myLights) + (3 * myHeavies) + (2 * myRanged);
    }

    public int getEnemyMilitaryStrength() {
        return enemyWorkers + (2 * enemyLights) + (3 * enemyHeavies) + (2 * enemyRanged);
    }

    static final class Builder {
        private int mapWidth = 8;
        private int mapHeight = 8;
        private int time = 0;
        private int myResources = 5;
        private int enemyResources = 5;
        private int myWorkers = 0;
        private int myLights = 0;
        private int myHeavies = 0;
        private int myRanged = 0;
        private int myBases = 0;
        private int myBarracks = 0;
        private int enemyWorkers = 0;
        private int enemyLights = 0;
        private int enemyHeavies = 0;
        private int enemyRanged = 0;
        private int enemyBases = 0;
        private int enemyBarracks = 0;
        private boolean pathToEnemyOpen = true;
        private int nearestEnemyToBase = 9999;
        private int nearbyResources = 0;

        Builder setMapWidth(int mapWidth) {
            this.mapWidth = mapWidth;
            return this;
        }

        Builder setMapHeight(int mapHeight) {
            this.mapHeight = mapHeight;
            return this;
        }

        Builder setTime(int time) {
            this.time = time;
            return this;
        }

        Builder setMyResources(int myResources) {
            this.myResources = myResources;
            return this;
        }

        Builder setEnemyResources(int enemyResources) {
            this.enemyResources = enemyResources;
            return this;
        }

        Builder setMyWorkers(int myWorkers) {
            this.myWorkers = myWorkers;
            return this;
        }

        Builder setMyLights(int myLights) {
            this.myLights = myLights;
            return this;
        }

        Builder setMyHeavies(int myHeavies) {
            this.myHeavies = myHeavies;
            return this;
        }

        Builder setMyRanged(int myRanged) {
            this.myRanged = myRanged;
            return this;
        }

        Builder setMyBases(int myBases) {
            this.myBases = myBases;
            return this;
        }

        Builder setMyBarracks(int myBarracks) {
            this.myBarracks = myBarracks;
            return this;
        }

        Builder setEnemyWorkers(int enemyWorkers) {
            this.enemyWorkers = enemyWorkers;
            return this;
        }

        Builder setEnemyLights(int enemyLights) {
            this.enemyLights = enemyLights;
            return this;
        }

        Builder setEnemyHeavies(int enemyHeavies) {
            this.enemyHeavies = enemyHeavies;
            return this;
        }

        Builder setEnemyRanged(int enemyRanged) {
            this.enemyRanged = enemyRanged;
            return this;
        }

        Builder setEnemyBases(int enemyBases) {
            this.enemyBases = enemyBases;
            return this;
        }

        Builder setEnemyBarracks(int enemyBarracks) {
            this.enemyBarracks = enemyBarracks;
            return this;
        }

        Builder setPathToEnemyOpen(boolean pathToEnemyOpen) {
            this.pathToEnemyOpen = pathToEnemyOpen;
            return this;
        }

        Builder setNearestEnemyToBase(int nearestEnemyToBase) {
            this.nearestEnemyToBase = nearestEnemyToBase;
            return this;
        }

        Builder setNearbyResources(int nearbyResources) {
            this.nearbyResources = nearbyResources;
            return this;
        }

        ChaseGameSnapshot build() {
            return new ChaseGameSnapshot(this);
        }
    }
}

final class DeterministicStrategyEngine {

    public MacroDecision decide(ChaseGameSnapshot snapshot, AdvisorRecommendation recommendation) {
        AdvisorRecommendation advice = AdvisorRecommendation.sanitize(recommendation);

        int mapSize = snapshot.getMapMaxDimension();
        int myMilitary = snapshot.getMyMilitaryStrength();
        int enemyMilitary = snapshot.getEnemyMilitaryStrength();
        int militaryLead = myMilitary - enemyMilitary;
        boolean underThreat = snapshot.getNearestEnemyToBase() <= 5;
        boolean blockedApproach = !snapshot.isPathToEnemyOpen();
        boolean earlyGame = snapshot.getTime() < 250;
        boolean hasBarracks = snapshot.getMyBarracks() > 0;

        double attackScore =
                (0.9 * militaryLead)
                        + (snapshot.getMyResources() >= 8 ? 1.0 : 0.0)
                        + (snapshot.getEnemyBases() == 0 ? 2.0 : 0.0)
                        + (blockedApproach ? -2.0 : 0.5)
                        + (underThreat ? -3.0 : 0.0)
                        + advice.getAttackBias();

        double economyScore =
                (snapshot.getMyWorkers() <= desiredWorkers(snapshot) ? 1.25 : -0.5)
                        + (hasBarracks ? -0.25 : 0.75)
                        + (snapshot.getNearbyResources() >= 3 ? 0.5 : 0.0)
                        + (underThreat ? -1.5 : 0.0)
                        + advice.getEconomyBias();

        BotPosture posture;
        if (underThreat || enemyMilitary > myMilitary + 2) {
            posture = BotPosture.DEFENSIVE;
        } else if (attackScore >= 2.0) {
            posture = BotPosture.AGGRESSIVE;
        } else {
            posture = BotPosture.BALANCED;
        }

        MacroStrategy strategy;
        UnitPreference unitPreference = UnitPreference.BALANCED;

        if (blockedApproach) {
            strategy = MacroStrategy.RANGED_RUSH;
            unitPreference = UnitPreference.RANGED;
        } else if (underThreat) {
            if (snapshot.getEnemyLights() > snapshot.getEnemyHeavies()) {
                strategy = MacroStrategy.HEAVY_RUSH;
                unitPreference = UnitPreference.HEAVY;
            } else {
                strategy = MacroStrategy.RANGED_RUSH;
                unitPreference = UnitPreference.RANGED;
            }
        } else if (earlyGame && snapshot.getMyWorkers() <= 2 && snapshot.getMyBarracks() == 0) {
            strategy = MacroStrategy.WORKER_RUSH;
        } else if (snapshot.getEnemyHeavies() > snapshot.getEnemyLights()) {
            strategy = MacroStrategy.RANGED_RUSH;
            unitPreference = UnitPreference.RANGED;
        } else if (snapshot.getEnemyRanged() > 0) {
            strategy = MacroStrategy.LIGHT_RUSH;
            unitPreference = UnitPreference.LIGHT;
        } else if (snapshot.getEnemyLights() >= snapshot.getEnemyHeavies() + 2) {
            strategy = MacroStrategy.HEAVY_RUSH;
            unitPreference = UnitPreference.HEAVY;
        } else if (mapSize >= 16 && hasBarracks) {
            strategy = MacroStrategy.LIGHT_RUSH;
        } else if (economyScore > 0.5 && snapshot.getMyBarracks() == 0) {
            strategy = MacroStrategy.WORKER_RUSH;
        } else {
            strategy = MacroStrategy.LIGHT_RUSH;
        }

        if (advice.getPreferredStrategy() != null && posture != BotPosture.DEFENSIVE) {
            strategy = advice.getPreferredStrategy();
        }

        if (advice.getUnitPreference() != null) {
            unitPreference = advice.getUnitPreference();
        }

        if (blockedApproach) {
            strategy = MacroStrategy.RANGED_RUSH;
            unitPreference = UnitPreference.RANGED;
        }
        if (posture == BotPosture.DEFENSIVE && strategy == MacroStrategy.WORKER_RUSH) {
            strategy = snapshot.getEnemyLights() > snapshot.getEnemyHeavies()
                    ? MacroStrategy.HEAVY_RUSH
                    : MacroStrategy.RANGED_RUSH;
        }
        if (unitPreference == null) {
            unitPreference = UnitPreference.BALANCED;
        }

        int harvestTarget = computeHarvestTarget(snapshot, posture);
        int barracksTarget = computeBarracksTarget(snapshot, posture, strategy);

        return new MacroDecision(
                strategy,
                posture,
                unitPreference,
                harvestTarget,
                barracksTarget,
                attackScore,
                economyScore);
    }

    private int desiredWorkers(ChaseGameSnapshot snapshot) {
        int desired = snapshot.getMapMaxDimension() <= 8 ? 2 : 3;
        if (snapshot.getMapMaxDimension() >= 24) {
            desired++;
        }
        return desired;
    }

    private int computeHarvestTarget(ChaseGameSnapshot snapshot, BotPosture posture) {
        int target = snapshot.getMapMaxDimension() <= 8 ? 1 : 2;
        if (snapshot.getNearbyResources() >= 3) {
            target++;
        }
        if (posture == BotPosture.DEFENSIVE) {
            target = Math.max(1, target - 1);
        }
        return Math.max(1, Math.min(4, target));
    }

    private int computeBarracksTarget(
            ChaseGameSnapshot snapshot,
            BotPosture posture,
            MacroStrategy strategy) {
        int target = 1;
        if (snapshot.getMapMaxDimension() >= 16) {
            target++;
        }
        if (strategy == MacroStrategy.RANGED_RUSH && snapshot.getMapMaxDimension() >= 24) {
            target++;
        }
        if (posture == BotPosture.DEFENSIVE) {
            target = Math.max(target, 1);
        }
        return Math.max(1, Math.min(3, target));
    }
}

final class MacroDecision {

    private final MacroStrategy strategy;
    private final BotPosture posture;
    private final UnitPreference unitPreference;
    private final int harvestTarget;
    private final int barracksTarget;
    private final double attackScore;
    private final double economyScore;

    MacroDecision(
            MacroStrategy strategy,
            BotPosture posture,
            UnitPreference unitPreference,
            int harvestTarget,
            int barracksTarget,
            double attackScore,
            double economyScore) {
        this.strategy = strategy;
        this.posture = posture;
        this.unitPreference = unitPreference;
        this.harvestTarget = harvestTarget;
        this.barracksTarget = barracksTarget;
        this.attackScore = attackScore;
        this.economyScore = economyScore;
    }

    public MacroStrategy getStrategy() {
        return strategy;
    }

    public BotPosture getPosture() {
        return posture;
    }

    public UnitPreference getUnitPreference() {
        return unitPreference;
    }

    public int getHarvestTarget() {
        return harvestTarget;
    }

    public int getBarracksTarget() {
        return barracksTarget;
    }

    public double getAttackScore() {
        return attackScore;
    }

    public double getEconomyScore() {
        return economyScore;
    }
}

enum MacroStrategy {
    WORKER_RUSH,
    LIGHT_RUSH,
    HEAVY_RUSH,
    RANGED_RUSH
}

class NoOpAdvisor implements StrategyAdvisor {

    @Override
    public AdvisorRecommendation advise(ChaseGameSnapshot snapshot) {
        return AdvisorRecommendation.neutral();
    }
}

class OllamaAdvisor implements StrategyAdvisor {

    private static final String OLLAMA_HOST =
            System.getenv().getOrDefault("OLLAMA_HOST", "http://localhost:11434");

    private final ChaseBotConfig config;
    private final Map<String, AdvisorRecommendation> cache;

    OllamaAdvisor(ChaseBotConfig config) {
        this.config = config;
        this.cache = new LinkedHashMap<String, AdvisorRecommendation>(16, 0.75f, true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<String, AdvisorRecommendation> eldest) {
                return size() > OllamaAdvisor.this.config.getAdvisorCacheEntries();
            }
        };
    }

    @Override
    public AdvisorRecommendation advise(ChaseGameSnapshot snapshot) throws Exception {
        if (!config.isAdvisorEnabled()) {
            return AdvisorRecommendation.neutral();
        }

        String cacheKey = buildCacheKey(snapshot);
        synchronized (cache) {
            if (cache.containsKey(cacheKey)) {
                return cache.get(cacheKey);
            }
        }

        String response = performRequest(snapshot);
        AdvisorRecommendation recommendation = parseRecommendation(response);
        synchronized (cache) {
            cache.put(cacheKey, recommendation);
        }
        return recommendation;
    }

    @Override
    public StrategyAdvisor copy() {
        return new OllamaAdvisor(config);
    }

    String performRequest(ChaseGameSnapshot snapshot) throws IOException {
        HttpURLConnection connection = (HttpURLConnection) new URL(OLLAMA_HOST + "/api/generate").openConnection();
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setConnectTimeout(config.getConnectTimeoutMs());
        connection.setReadTimeout(config.getReadTimeoutMs());
        connection.setRequestProperty("Content-Type", "application/json");

        JsonObject payload = new JsonObject();
        payload.addProperty("model", config.getOllamaModel());
        payload.addProperty("prompt", buildPrompt(snapshot));
        payload.addProperty("stream", false);
        payload.addProperty("format", "json");

        JsonObject options = new JsonObject();
        options.addProperty("temperature", 0.2);
        payload.add("options", options);

        byte[] requestBody = payload.toString().getBytes(StandardCharsets.UTF_8);
        try (OutputStream output = connection.getOutputStream()) {
            output.write(requestBody);
        }

        int status = connection.getResponseCode();
        BufferedReader reader = new BufferedReader(new InputStreamReader(
                status >= 200 && status < 300 ? connection.getInputStream() : connection.getErrorStream(),
                StandardCharsets.UTF_8));

        StringBuilder sb = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            sb.append(line);
        }
        reader.close();
        connection.disconnect();

        if (status < 200 || status >= 300) {
            throw new IOException("Ollama advisor request failed with status " + status + ": " + sb);
        }
        return sb.toString();
    }

    AdvisorRecommendation parseRecommendation(String responseBody) {
        if (responseBody == null || responseBody.isBlank()) {
            return AdvisorRecommendation.neutral();
        }

        JsonObject root = JsonParser.parseString(responseBody).getAsJsonObject();
        String payload = root.has("response") ? root.get("response").getAsString() : null;
        if ((payload == null || payload.isBlank()) && root.has("message")) {
            JsonObject message = root.getAsJsonObject("message");
            if (message.has("content")) {
                payload = message.get("content").getAsString();
            }
        }
        if (payload == null || payload.isBlank()) {
            return AdvisorRecommendation.neutral();
        }

        payload = extractJsonObject(payload);
        if (payload == null || payload.isBlank()) {
            return AdvisorRecommendation.neutral();
        }

        JsonObject json = JsonParser.parseString(payload).getAsJsonObject();
        double attackBias = json.has("attack_bias") ? json.get("attack_bias").getAsDouble() : 0.0;
        double economyBias = json.has("economy_bias") ? json.get("economy_bias").getAsDouble() : 0.0;
        MacroStrategy strategy = parseStrategy(json.has("preferred_strategy")
                ? json.get("preferred_strategy").getAsString()
                : null);
        UnitPreference unitPreference = parseUnitPreference(json.has("unit_preference")
                ? json.get("unit_preference").getAsString()
                : null);

        return AdvisorRecommendation.of(attackBias, economyBias, strategy, unitPreference);
    }

    private String extractJsonObject(String payload) {
        String trimmed = payload.trim();
        if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
            return trimmed;
        }
        int start = trimmed.indexOf('{');
        int end = trimmed.lastIndexOf('}');
        if (start >= 0 && end > start) {
            return trimmed.substring(start, end + 1);
        }
        return null;
    }

    private String buildPrompt(ChaseGameSnapshot snapshot) {
        return "You are advising a deterministic microRTS bot running with Ollama. "
                + "Return exactly one JSON object and nothing else. "
                + "Use conservative, practical nudges only.\n"
                + "Schema:{"
                + "\"attack_bias\":number,"
                + "\"economy_bias\":number,"
                + "\"preferred_strategy\":\"WORKER_RUSH|LIGHT_RUSH|HEAVY_RUSH|RANGED_RUSH|BALANCED\","
                + "\"unit_preference\":\"BALANCED|LIGHT|HEAVY|RANGED\""
                + "}\n"
                + "Map=" + snapshot.getMapWidth() + "x" + snapshot.getMapHeight()
                + ", time=" + snapshot.getTime()
                + ", myResources=" + snapshot.getMyResources()
                + ", enemyResources=" + snapshot.getEnemyResources()
                + ", myUnits=[workers:" + snapshot.getMyWorkers()
                + ", light:" + snapshot.getMyLights()
                + ", heavy:" + snapshot.getMyHeavies()
                + ", ranged:" + snapshot.getMyRanged()
                + ", bases:" + snapshot.getMyBases()
                + ", barracks:" + snapshot.getMyBarracks() + "]"
                + ", enemyUnits=[workers:" + snapshot.getEnemyWorkers()
                + ", light:" + snapshot.getEnemyLights()
                + ", heavy:" + snapshot.getEnemyHeavies()
                + ", ranged:" + snapshot.getEnemyRanged()
                + ", bases:" + snapshot.getEnemyBases()
                + ", barracks:" + snapshot.getEnemyBarracks() + "]"
                + ", nearestEnemyToBase=" + snapshot.getNearestEnemyToBase()
                + ", pathOpen=" + snapshot.isPathToEnemyOpen()
                + ", nearbyResources=" + snapshot.getNearbyResources()
                + ". Clamp attack_bias and economy_bias to [-3,3]. "
                + "If unsure, return BALANCED preferences and 0 biases.";
    }

    private String buildCacheKey(ChaseGameSnapshot snapshot) {
        return snapshot.getMapWidth()
                + "x" + snapshot.getMapHeight()
                + ":" + snapshot.getTime() / Math.max(1, config.getConsultationInterval())
                + ":" + snapshot.getMyResources()
                + ":" + snapshot.getMyMilitaryStrength()
                + ":" + snapshot.getEnemyMilitaryStrength()
                + ":" + snapshot.getNearestEnemyToBase()
                + ":" + snapshot.isPathToEnemyOpen();
    }

    private MacroStrategy parseStrategy(String raw) {
        if (raw == null || raw.isBlank() || "BALANCED".equalsIgnoreCase(raw)) {
            return null;
        }
        try {
            return MacroStrategy.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException ex) {
            return null;
        }
    }

    private UnitPreference parseUnitPreference(String raw) {
        if (raw == null || raw.isBlank()) {
            return null;
        }
        try {
            return UnitPreference.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException ex) {
            return null;
        }
    }
}

interface StrategyAdvisor {

    AdvisorRecommendation advise(ChaseGameSnapshot snapshot) throws Exception;

    default StrategyAdvisor copy() {
        return this;
    }
}

enum UnitPreference {
    BALANCED,
    LIGHT,
    HEAVY,
    RANGED
}
