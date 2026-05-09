package ai.abstraction.submissions.adil_bot;

import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.AI;
import ai.core.ParameterSpecification;
import java.util.*;
import rts.*;
import rts.units.*;

public class MyBot extends AbstractionLayerAI {

    Random r = new Random();
    protected UnitTypeTable utt;

    UnitType workerType;
    UnitType baseType;
    UnitType barracksType;
    UnitType lightType;

    public MyBot(UnitTypeTable a_utt) {
        super(new AStarPathFinding());
        reset(a_utt);
    }

    public MyBot(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        workerType = utt.getUnitType("Worker");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
        lightType = utt.getUnitType("Light");
    }

    @Override
    public AI clone() {
        return new MyBot(utt);
    }

    @Override
    public PlayerAction getAction(int player, GameState gs) {

        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);

        List<Unit> workers = new ArrayList<>();
        List<Unit> lights = new ArrayList<>();

        for (Unit u : pgs.getUnits()) {
            if (u.getPlayer() == player) {
                if (u.getType() == workerType) workers.add(u);
                if (u.getType() == lightType) lights.add(u);
            }
        }

        // ===== BASE =====
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player && gs.getActionAssignment(u) == null) {
                if (workers.size() < 3 && p.getResources() >= workerType.cost) {
                    train(u, workerType);
                }
            }
        }

        // ===== BARRACKS =====
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == barracksType && u.getPlayer() == player && gs.getActionAssignment(u) == null) {
                if (p.getResources() >= lightType.cost) {
                    train(u, lightType);
                }
            }
        }

        // ===== BUILD STRUCTURES =====
        int bases = 0, barracks = 0;

        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType && u.getPlayer() == player) bases++;
            if (u.getType() == barracksType && u.getPlayer() == player) barracks++;
        }

        List<Unit> freeWorkers = new ArrayList<>(workers);

        if (bases == 0 && !freeWorkers.isEmpty()) {
            Unit w = freeWorkers.remove(0);
            if (p.getResources() >= baseType.cost) {
                build(w, baseType, w.getX(), w.getY());
            }
        }

        if (barracks == 0 && !freeWorkers.isEmpty()) {
            Unit w = freeWorkers.remove(0);
            if (p.getResources() >= barracksType.cost) {
                build(w, barracksType, w.getX(), w.getY());
            }
        }

        // ===== WORKERS: HARVEST =====
        for (Unit w : freeWorkers) {
            Unit closestResource = null;
            Unit closestBase = null;
            int d1 = Integer.MAX_VALUE;
            int d2 = Integer.MAX_VALUE;

            for (Unit u : pgs.getUnits()) {
                if (u.getType().isResource) {
                    int d = dist(w, u);
                    if (d < d1) {
                        closestResource = u;
                        d1 = d;
                    }
                }
                if (u.getType().isStockpile && u.getPlayer() == player) {
                    int d = dist(w, u);
                    if (d < d2) {
                        closestBase = u;
                        d2 = d;
                    }
                }
            }

            if (closestResource != null && closestBase != null) {
                harvest(w, closestResource, closestBase);
            }
        }

        // ===== LIGHT UNITS: ATTACK =====
        for (Unit u : lights) {
            if (gs.getActionAssignment(u) == null) {
                Unit enemy = closestEnemy(u, pgs, player);
                if (enemy != null) attack(u, enemy);
            }
        }

        // ===== WORKER RUSH (EARLY GAME) =====
        if (gs.getTime() < 200) {
            for (Unit w : workers) {
                Unit enemy = closestEnemy(w, pgs, player);
                if (enemy != null) attack(w, enemy);
            }
        }

        return translateActions(player, gs);
    }

    private Unit closestEnemy(Unit u, PhysicalGameState pgs, int player) {
        Unit closest = null;
        int best = Integer.MAX_VALUE;

        for (Unit e : pgs.getUnits()) {
            if (e.getPlayer() >= 0 && e.getPlayer() != player) {
                int d = dist(u, e);
                if (d < best) {
                    best = d;
                    closest = e;
                }
            }
        }
        return closest;
    }

    private int dist(Unit a, Unit b) {
        return Math.abs(a.getX() - b.getX()) + Math.abs(a.getY() - b.getY());
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }
}