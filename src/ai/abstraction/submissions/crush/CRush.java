/*
 * Adaptive ranged rush bot with tactical formations.
 * Worker rush on small maps, ranged units with kiting and
 * square formation tactics on larger maps.
 *
 * @author Cristiano D'Angelo
 * Submitted by Chase Morris (PR #113)
 */
package ai.abstraction.submissions.crush;

import ai.abstraction.AbstractAction;
import ai.abstraction.AbstractionLayerAI;
import ai.abstraction.Attack;
import ai.abstraction.Harvest;
import ai.abstraction.pathfinding.AStarPathFinding;
import ai.core.AI;
import ai.abstraction.pathfinding.PathFinding;
import ai.core.ParameterSpecification;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Random;
import rts.GameState;
import rts.PhysicalGameState;
import rts.Player;
import rts.PlayerAction;
import rts.ResourceUsage;
import rts.UnitAction;
import rts.units.*;
import util.XMLWriter;

public class CRush extends AbstractionLayerAI {

    Random r = new Random();
    protected UnitTypeTable utt;
    UnitType workerType;
    UnitType baseType;
    UnitType barracksType;
    UnitType rangedType;
    UnitType heavyType;
    UnitType lightType;
    boolean buildingRacks = false;
    int resourcesUsed = 0;
    boolean isRush = false;

    public CRush(UnitTypeTable a_utt) {
        this(a_utt, new AStarPathFinding());
    }

    public CRush(UnitTypeTable a_utt, PathFinding a_pf) {
        super(a_pf);
        reset(a_utt);
    }

    public void reset() {
        super.reset();
    }

    public void reset(UnitTypeTable a_utt) {
        utt = a_utt;
        workerType = utt.getUnitType("Worker");
        baseType = utt.getUnitType("Base");
        barracksType = utt.getUnitType("Barracks");
        rangedType = utt.getUnitType("Ranged");
        lightType = utt.getUnitType("Light");
    }

    public AI clone() {
        return new CRush(utt, pf);
    }

    public PlayerAction getAction(int player, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Player p = gs.getPlayer(player);

        if ((pgs.getWidth() * pgs.getHeight()) <= 144) {
            isRush = true;
        }

        List<Unit> workers = new LinkedList<>();
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canHarvest
                    && u.getPlayer() == player) {
                workers.add(u);
            }
        }
        if (isRush) {
            rushWorkersBehavior(workers, p, pgs, gs);
        } else {
            workersBehavior(workers, p, pgs, gs);
        }

        // behavior of bases:
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == baseType
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {

                if (isRush) {
                    rushBaseBehavior(u, p, pgs);
                } else {
                    baseBehavior(u, p, pgs, gs);
                }
            }
        }

        // behavior of barracks:
        for (Unit u : pgs.getUnits()) {
            if (u.getType() == barracksType
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                barracksBehavior(u, p, pgs);
            }
        }

        // behavior of melee units:
        for (Unit u : pgs.getUnits()) {
            if (u.getType().canAttack && !u.getType().canHarvest
                    && u.getPlayer() == player
                    && gs.getActionAssignment(u) == null) {
                if (u.getType() == rangedType) {
                    rangedUnitBehavior(u, p, gs);
                } else {
                    meleeUnitBehavior(u, p, gs);
                }
            }
        }

        return translateActions(player, gs);
    }

    public void baseBehavior(Unit u, Player p, PhysicalGameState pgs, GameState gs) {

        int nbases = 0;
        int nbarracks = 0;
        int nworkers = 0;
        int nranged = 0;
        int resources = p.getResources();

        for (Unit u2 : pgs.getUnits()) {
            if (u2.getType() == workerType
                    && u2.getPlayer() == p.getID()) {
                nworkers++;
            }
            if (u2.getType() == barracksType
                    && u2.getPlayer() == p.getID()) {
                nbarracks++;
            }
            if (u2.getType() == baseType
                    && u2.getPlayer() == p.getID()) {
                nbases++;
            }
            if (u2.getType() == rangedType
                    && u2.getPlayer() == p.getID()) {
                nranged++;
            }
        }
        if ((nworkers < (nbases + 1) && p.getResources() >= workerType.cost) || nranged > 6) {
            train(u, workerType);
        }

        //Buffers the resources that are being used for barracks
        if (resourcesUsed != barracksType.cost * nbarracks) {
            resources = resources - barracksType.cost;
        }

        if (buildingRacks && (resources >= workerType.cost + rangedType.cost)) {
            train(u, workerType);
        }
    }

    public void barracksBehavior(Unit u, Player p, PhysicalGameState pgs) {
        if (p.getResources() >= rangedType.cost) {
            train(u, rangedType);
        }
    }

    public void meleeUnitBehavior(Unit u, Player p, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit closestEnemy = null;
        Unit closestRacks = null;
        Unit closestBase = null;
        Unit closestEnemyBase = null;
        int closestDistance = 0;
        for (Unit u2 : pgs.getUnits()) {
            if (u2.getPlayer() >= 0 && u2.getPlayer() != p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if (closestEnemy == null || d < closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }
            }
            if (u2.getType() == barracksType && u2.getPlayer() == p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if (closestRacks == null || d < closestDistance) {
                    closestRacks = u2;
                    closestDistance = d;
                }
            }
            if (u2.getType() == baseType && u2.getPlayer() == p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if (closestBase == null || d < closestDistance) {
                    closestBase = u2;
                    closestDistance = d;
                }
            }
            if (u2.getType() == baseType && u2.getPlayer() != p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if (closestEnemyBase == null || d < closestDistance) {
                    closestEnemyBase = u2;
                    closestDistance = d;
                }
            }

        }
        if (closestEnemy != null) {
            if (gs.getTime() < 400 || isRush) {
                attack(u, closestEnemy);
            } else {
                rangedTactic(u, closestEnemy, closestBase, closestEnemyBase, utt, p);
            }
        }
    }

    public void rangedUnitBehavior(Unit u, Player p, GameState gs) {
        PhysicalGameState pgs = gs.getPhysicalGameState();
        Unit closestEnemy = null;
        Unit closestRacks = null;
        Unit closestBase = null;
        Unit closestEnemyBase = null;
        int closestDistance = 0;

        for (Unit u2 : pgs.getUnits()) {
            if (u2.getPlayer() >= 0 && u2.getPlayer() != p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if (closestEnemy == null || d < closestDistance) {
                    closestEnemy = u2;
                    closestDistance = d;
                }
            }
            if (u2.getType() == baseType && u2.getPlayer() == p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if (closestBase == null || d < closestDistance) {
                    closestBase = u2;
                    closestDistance = d;
                }
            }

            if (u2.getType() == barracksType && u2.getPlayer() == p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if (closestRacks == null || d < closestDistance) {
                    closestRacks = u2;
                    closestDistance = d;
                }
            }
            if (u2.getType() == baseType && u2.getPlayer() != p.getID()) {
                int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                if (closestEnemyBase == null || d < closestDistance) {
                    closestEnemyBase = u2;
                    closestDistance = d;
                }
            }
        }
        if (closestEnemy != null) {
            rangedTactic(u, closestEnemy, closestBase, closestEnemyBase, utt, p);
        }
    }

    public void workersBehavior(List<Unit> workers, Player p, PhysicalGameState pgs, GameState gs) {
        int nbases = 0;
        int nbarracks = 0;
        int nworkers = 0;
        resourcesUsed = 0;

        List<Unit> freeWorkers = new LinkedList<>();
        List<Unit> battleWorkers = new LinkedList<>();

        for (Unit u2 : pgs.getUnits()) {
            if (u2.getType() == baseType
                    && u2.getPlayer() == p.getID()) {
                nbases++;
            }
            if (u2.getType() == barracksType
                    && u2.getPlayer() == p.getID()) {
                nbarracks++;
            }
            if (u2.getType() == workerType
                    && u2.getPlayer() == p.getID()) {
                nworkers++;
            }
        }

        if (workers.size() > (nbases + 1)) {
            for (int n = 0; n < (nbases + 1); n++) {
                freeWorkers.add(workers.get(0));
                workers.remove(0);
            }
            battleWorkers.addAll(workers);
        } else {
            freeWorkers.addAll(workers);
        }

        if (workers.isEmpty()) {
            return;
        }

        List<Integer> reservedPositions = new LinkedList<>();
        if (nbases == 0 && !freeWorkers.isEmpty()) {
            // build a base:
            if (p.getResources() >= baseType.cost) {
                Unit u = freeWorkers.remove(0);
                buildIfNotAlreadyBuilding(u, baseType, u.getX(), u.getY(), reservedPositions, p, pgs);
            }
        }
        if ((nbarracks == 0) && (!freeWorkers.isEmpty()) && nworkers > 1
                && p.getResources() >= barracksType.cost) {

            int resources = p.getResources();
            Unit u = freeWorkers.remove(0);
            buildIfNotAlreadyBuilding(u, barracksType, u.getX(), u.getY(), reservedPositions, p, pgs);
            resourcesUsed += barracksType.cost;
            buildingRacks = true;

        } else {
            resourcesUsed = barracksType.cost * nbarracks;
        }

        if (nbarracks > 1) {
            buildingRacks = true;
        }

        for (Unit u : battleWorkers) {
            meleeUnitBehavior(u, p, gs);
        }

        // harvest with all the free workers:
        for (Unit u : freeWorkers) {
            Unit closestBase = null;
            Unit closestResource = null;
            Unit closestEnemyBase = null;
            int closestDistance = 0;
            for (Unit u2 : pgs.getUnits()) {
                if (u2.getType().isResource) {
                    int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                    if (closestResource == null || d < closestDistance) {
                        closestResource = u2;
                        closestDistance = d;
                    }
                }
                if (u2.getType() == baseType && u2.getPlayer() != p.getID()) {
                    int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                    if (closestEnemyBase == null || d < closestDistance) {
                        closestEnemyBase = u2;
                        closestDistance = d;
                    }
                }
            }
            closestDistance = 0;
            for (Unit u2 : pgs.getUnits()) {
                if (u2.getType().isStockpile && u2.getPlayer() == p.getID()) {
                    int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                    if (closestBase == null || d < closestDistance) {
                        closestBase = u2;
                        closestDistance = d;
                    }
                }
            }
            if (closestResource == null || unitDistance(closestResource, closestEnemyBase) < unitDistance(closestResource, closestBase)) {
                //Do nothing - resource is closer to enemy
            } else {
                if (closestResource != null && closestBase != null) {
                    AbstractAction aa = getAbstractAction(u);
                    if (aa instanceof Harvest) {
                        Harvest h_aa = (Harvest) aa;

                        if (h_aa.getTarget() != closestResource || h_aa.getBase() != closestBase) {
                            harvest(u, closestResource, closestBase);
                        }
                    } else {
                        harvest(u, closestResource, closestBase);
                    }
                }
            }
        }
    }

    public void rushBaseBehavior(Unit u, Player p, PhysicalGameState pgs) {
        if (p.getResources() >= workerType.cost) {
            train(u, workerType);
        }
    }

    public void rushWorkersBehavior(List<Unit> workers, Player p, PhysicalGameState pgs, GameState gs) {
        int nbases = 0;
        int nworkers = 0;
        resourcesUsed = 0;

        List<Unit> freeWorkers = new LinkedList<>();
        List<Unit> battleWorkers = new LinkedList<>();

        for (Unit u2 : pgs.getUnits()) {
            if (u2.getType() == baseType
                    && u2.getPlayer() == p.getID()) {
                nbases++;
            }
            if (u2.getType() == workerType
                    && u2.getPlayer() == p.getID()) {
                nworkers++;
            }
        }
        if (p.getResources() == 0) {
            battleWorkers.addAll(workers);
        } else if (workers.size() > (nbases)) {
            for (int n = 0; n < (nbases); n++) {
                freeWorkers.add(workers.get(0));
                workers.remove(0);
            }
            battleWorkers.addAll(workers);
        } else {
            freeWorkers.addAll(workers);
        }

        if (workers.isEmpty()) {
            return;
        }

        List<Integer> reservedPositions = new LinkedList<>();
        if (nbases == 0 && !freeWorkers.isEmpty()) {
            // build a base:
            if (p.getResources() >= baseType.cost) {
                Unit u = freeWorkers.remove(0);
                buildIfNotAlreadyBuilding(u, baseType, u.getX(), u.getY(), reservedPositions, p, pgs);
            }
        }

        for (Unit u : battleWorkers) {
            meleeUnitBehavior(u, p, gs);
        }

        // harvest with all the free workers:
        for (Unit u : freeWorkers) {
            Unit closestBase = null;
            Unit closestResource = null;
            int closestDistance = 0;
            for (Unit u2 : pgs.getUnits()) {
                if (u2.getType().isResource) {
                    int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                    if (closestResource == null || d < closestDistance) {
                        closestResource = u2;
                        closestDistance = d;
                    }
                }
            }
            closestDistance = 0;
            for (Unit u2 : pgs.getUnits()) {
                if (u2.getType().isStockpile && u2.getPlayer() == p.getID()) {
                    int d = Math.abs(u2.getX() - u.getX()) + Math.abs(u2.getY() - u.getY());
                    if (closestBase == null || d < closestDistance) {
                        closestBase = u2;
                        closestDistance = d;
                    }
                }
            }
            if (closestResource != null && closestBase != null) {
                AbstractAction aa = getAbstractAction(u);
                if (aa instanceof Harvest) {
                    Harvest h_aa = (Harvest) aa;
                    if (h_aa.getTarget() != closestResource || h_aa.getBase() != closestBase) {
                        harvest(u, closestResource, closestBase);
                    }
                } else {
                    harvest(u, closestResource, closestBase);
                }
            }
        }
    }

    public void rangedTactic(Unit u, Unit target, Unit home, Unit enemyBase, UnitTypeTable utt, Player p) {
        actions.put(u, new CRangedTactic(u, target, home, enemyBase, pf, utt, p));
    }

    //Calculates distance between unit a and unit b
    public double unitDistance(Unit a, Unit b) {
        if (a == null || b == null) {
            return 0.0;
        }
        int dx = b.getX() - a.getX();
        int dy = b.getY() - a.getY();
        return Math.sqrt(dx * dx + dy * dy);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        List<ParameterSpecification> parameters = new ArrayList<>();
        parameters.add(new ParameterSpecification("PathFinding", PathFinding.class, new AStarPathFinding()));
        return parameters;
    }

    // Inner class for ranged tactical behavior
    static class CRangedTactic extends AbstractAction {

        Unit target;
        PathFinding pf;
        Unit home;
        Unit enemyBase;
        UnitType workerType;
        UnitType rangedType;
        UnitType heavyType;
        UnitType baseType;
        UnitType barracksType;
        UnitType resourceType;
        UnitType lightType;
        UnitTypeTable utt;
        Player p;

        public CRangedTactic(Unit u, Unit a_target, Unit h, Unit eb, PathFinding a_pf, UnitTypeTable ut, Player pl) {
            super(u);
            target = a_target;
            pf = a_pf;
            home = h;
            utt = ut;
            p = pl;
            workerType = utt.getUnitType("Worker");
            rangedType = utt.getUnitType("Ranged");
            heavyType = utt.getUnitType("Heavy");
            baseType = utt.getUnitType("Base");
            barracksType = utt.getUnitType("Barracks");
            resourceType = utt.getUnitType("Resource");
            lightType = utt.getUnitType("Light");
            enemyBase = eb;
        }

        public boolean completed(GameState gs) {
            PhysicalGameState pgs = gs.getPhysicalGameState();
            return !pgs.getUnits().contains(target);
        }

        public boolean equals(Object o) {
            if (!(o instanceof CRangedTactic)) {
                return false;
            }
            CRangedTactic a = (CRangedTactic) o;
            return target.getID() == a.target.getID() && pf.getClass() == a.pf.getClass();
        }

        public void toxml(XMLWriter w) {
            w.tagWithAttributes("Attack", "unitID=\"" + getUnit().getID() + "\" target=\"" + target.getID() + "\" pathfinding=\"" + pf.getClass().getSimpleName() + "\"");
            w.tag("/Attack");
        }

        public UnitAction execute(GameState gs, ResourceUsage ru) {
            PhysicalGameState pgs = gs.getPhysicalGameState();

            Unit unit = getUnit();

            boolean timeToAttack = false;

            if (home == null) {
                home = unit;
            }

            if (enemyBase == null) {
                enemyBase = target;
            }

            //Determining distances
            double rd = 0.0;

            if (home != null) {
                rd = distance(unit, home);
            }

            double d = distance(unit, target);

            //Counting enemy units
            List<Unit> gameUnits = pgs.getUnits();

            int nEnemyBases = 0;
            int enemyAttackUnits = 0;
            int enemyWorkers = 0;
            int cutoffTime = 5000;

            if ((pgs.getWidth() * pgs.getHeight()) > 3000) {
                cutoffTime = 15000;
            }

            for (Unit u2 : gameUnits) {
                if (u2 != null && u2.getPlayer() != p.getID() && u2.getType() == baseType) {
                    nEnemyBases++;
                }

                if (u2 != null && u2.getPlayer() != p.getID()
                        && (u2.getType() == rangedType || u2.getType() == heavyType || u2.getType() == lightType)) {
                    enemyAttackUnits++;
                }

                if (u2 != null && u2.getPlayer() != p.getID() && u2.getType() == workerType) {
                    enemyWorkers++;
                }
            }

            //Determining if its time to attack
            if ((enemyWorkers < (2 * nEnemyBases) || nEnemyBases == 0) && enemyAttackUnits == 0) {
                timeToAttack = true;
            }

            if (gs.getTime() > cutoffTime) {
                timeToAttack = true;
            }

            //Finding ranged ally and distance from ally to target
            Unit ally = nearestRangedAlly(enemyBase, gameUnits, gs);

            double ad = 0.0;

            if (ally != null) {
                ad = distance(ally, target);
            }

            //Action for workers
            if (unit.getType() == workerType) {
                UnitAction move = null;
                if (d <= unit.getAttackRange()) {
                    return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, target.getX(), target.getY());
                } else if (timeToAttack) {
                    move = pf.findPathToPositionInRange(unit, target.getX() + target.getY() * gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
                } else if (ally != null) {

                    if (d > ad) {
                        move = pf.findPathToPositionInRange(unit, target.getX() + target.getY() * gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
                    } else {
                        move = pf.findPathToPositionInRange(unit, ally.getX() + ally.getY() * gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
                    }
                    if (move == null) {
                        move = pf.findPathToPositionInRange(unit, (ally.getX() - 1) + (ally.getY()) * gs.getPhysicalGameState().getWidth(), unit.getAttackRange() + 1, gs, ru);
                    }
                    if (move == null) {
                        move = pf.findPathToPositionInRange(unit, target.getX() + target.getY() * gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
                    }
                } else {
                    move = pf.findPathToPositionInRange(unit, target.getX() + target.getY() * gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
                }

                if (move != null && gs.isUnitActionAllowed(unit, move)) {
                    return move;
                }
                return null;
            }

            //Action for ranged units
            if (d <= unit.getAttackRange()) {
                return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, target.getX(), target.getY());
            } //If the unit is the ally closest to enemy base
            else if ((ally == null || ally.getID() == unit.getID())) {
                UnitAction move = null;

                if (timeToAttack && (target.getType() == baseType)) {
                    move = pf.findPathToPositionInRange(unit, target.getX() + target.getY() * gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
                } else if (rd < 5 || (distance(unit, enemyBase) > distance(home, enemyBase))) {
                    move = pf.findPathToPositionInRange(unit, enemyBase.getX() + enemyBase.getY() * gs.getPhysicalGameState().getWidth(), unit.getAttackRange(), gs, ru);
                }
                if (move != null && gs.isUnitActionAllowed(unit, move)) {
                    return move;
                }
                return null;
            } else if (timeToAttack) {

                //Attack behavior
                if (d <= (unit.getAttackRange()) - 1 && rd > 2 && unit.getMoveTime() < target.getMoveTime()) {
                    UnitAction move = pf.findPathToPositionInRange(unit, home.getX() + home.getY() * gs.getPhysicalGameState().getWidth(), getUnit().getAttackRange(), gs, ru);
                    if (move != null && gs.isUnitActionAllowed(unit, move)) {
                        return move;
                    }
                    return null;
                } else if (d <= unit.getAttackRange()) {
                    return new UnitAction(UnitAction.TYPE_ATTACK_LOCATION, target.getX(), target.getY());
                } else {
                    // move towards the unit:
                    UnitAction move = pf.findPathToPositionInRange(unit, target.getX() + target.getY() * gs.getPhysicalGameState().getWidth(), getUnit().getAttackRange(), gs, ru);
                    if (move != null && gs.isUnitActionAllowed(unit, move)) {
                        return move;
                    }
                    return null;
                }

            } //Behavior for ranged units to move into a position next to the leading ranged unit (ally)
            else {

                Unit atUp = pgs.getUnitAt(ally.getX(), ally.getY() - 1);
                Unit atUpLeft = pgs.getUnitAt(ally.getX() - 1, ally.getY() - 1);
                Unit atLeft = pgs.getUnitAt(ally.getX() - 1, ally.getY());
                Unit atDown = pgs.getUnitAt(ally.getX(), ally.getY() + 1);
                Unit atDownRight = pgs.getUnitAt(ally.getX() + 1, ally.getY() + 1);
                Unit atRight = pgs.getUnitAt(ally.getX() + 1, ally.getY());

                boolean positionFound = false;

                if ((coordDistance(ally.getX(), (ally.getY() + 1), enemyBase.getX(), enemyBase.getY()) > distance(ally, enemyBase))) {
                    while (!positionFound) {
                        if ((atDown != null && unit != atDown) && (atDownRight != null && unit != atDownRight)
                                && (atRight != null && unit != atRight)) {
                            ally = atRight;
                        } else {
                            positionFound = true;
                        }

                        atDown = pgs.getUnitAt(ally.getX(), ally.getY() + 1);
                        atDownRight = pgs.getUnitAt(ally.getX() + 1, ally.getY() + 1);
                        atRight = pgs.getUnitAt(ally.getX() + 1, ally.getY());

                        if (atDown == null || atDownRight == null || atRight == null) {
                            positionFound = true;
                        }

                    }
                } else {
                    while (!positionFound) {

                        if ((atUp != null && unit != atUp) && (atUpLeft != null && unit != atUpLeft)
                                && (atLeft != null && unit != atLeft)) {
                            ally = atLeft;
                        } else {
                            positionFound = true;
                        }

                        atUp = pgs.getUnitAt(ally.getX(), ally.getY() - 1);
                        atUpLeft = pgs.getUnitAt(ally.getX() - 1, ally.getY() - 1);
                        atLeft = pgs.getUnitAt(ally.getX() - 1, ally.getY());

                        if (atUp == null || atUpLeft == null || atLeft == null) {
                            positionFound = true;
                        }

                    }
                }
                return squareMove(gs, ru, ally);

            }
        }

        //Calculates distance between unit a and unit b
        public double distance(Unit a, Unit b) {
            if (a == null || b == null) {
                return 0.0;
            }
            int dx = b.getX() - a.getX();
            int dy = b.getY() - a.getY();
            return Math.sqrt(dx * dx + dy * dy);
        }

        //Calculates distance between positions a and b using x,y coordinates
        public double coordDistance(int xa, int ya, int xb, int yb) {
            int dx = xb - xa;
            int dy = yb - ya;
            return Math.sqrt(dx * dx + dy * dy);
        }

        //Figures out correct move action for a square unit formation
        public UnitAction squareMove(GameState gs, ResourceUsage ru, Unit targetUnit) {
            PhysicalGameState pgs = gs.getPhysicalGameState();
            Unit unit = getUnit();
            Unit ally = targetUnit;

            Unit atUp = pgs.getUnitAt(ally.getX(), ally.getY() - 1);
            Unit atUpLeft = pgs.getUnitAt(ally.getX() - 1, ally.getY() - 1);
            Unit atLeft = pgs.getUnitAt(ally.getX() - 1, ally.getY());
            Unit atDown = pgs.getUnitAt(ally.getX(), ally.getY() + 1);
            Unit atDownRight = pgs.getUnitAt(ally.getX() + 1, ally.getY() + 1);
            Unit atRight = pgs.getUnitAt(ally.getX() + 1, ally.getY());

            UnitAction moveToUp = pf.findPath(unit, (ally.getX()) + (ally.getY() - 1) * gs.getPhysicalGameState().getWidth(), gs, ru);
            UnitAction moveToUpLeft = pf.findPath(unit, (ally.getX() - 1) + (ally.getY() - 1) * gs.getPhysicalGameState().getWidth(), gs, ru);
            UnitAction moveToLeft = pf.findPath(unit, (ally.getX() - 1) + (ally.getY()) * gs.getPhysicalGameState().getWidth(), gs, ru);
            UnitAction moveToDown = pf.findPath(unit, (ally.getX()) + (ally.getY() + 1) * gs.getPhysicalGameState().getWidth(), gs, ru);
            UnitAction moveToDownRight = pf.findPath(unit, (ally.getX() + 1) + (ally.getY() + 1) * gs.getPhysicalGameState().getWidth(), gs, ru);
            UnitAction moveToRight = pf.findPath(unit, (ally.getX() + 1) + (ally.getY()) * gs.getPhysicalGameState().getWidth(), gs, ru);

            if (coordDistance(ally.getX(), (ally.getY() + 1), enemyBase.getX(), enemyBase.getY()) > distance(ally, enemyBase)) {
                UnitAction move = null;
                if (unit == atDown || unit == atDownRight || unit == atRight) {
                    return null;
                }
                if (atDown == null) {
                    move = moveToDown;
                } else if (atRight == null) {
                    move = moveToRight;
                } else if (atDownRight == null) {
                    move = moveToDownRight;
                }

                if (move != null && gs.isUnitActionAllowed(unit, move)) {
                    return move;
                }
                return null;
            } else {
                UnitAction move = null;

                if (unit == atUp || unit == atUpLeft || unit == atLeft) {
                    return null;
                }
                if (atUp == null) {
                    move = moveToUp;
                } else if (atLeft == null) {
                    move = moveToLeft;
                } else if (atUpLeft == null) {
                    move = moveToUpLeft;
                }

                if (move != null && gs.isUnitActionAllowed(unit, move)) {
                    return move;
                }
                return null;
            }
        }

        //Finds nearest ranged unit from starting point
        public Unit nearestRangedAlly(Unit start, List<Unit> units, GameState gs) {
            Unit nearestUnit = null;
            double nearestDistance = -1;

            if (start != null) {
                for (Unit u2 : units) {
                    if (u2 != null && u2.getPlayer() == p.getID() && u2.getType() == rangedType) {

                        int dx = start.getX() - u2.getX();
                        int dy = start.getY() - u2.getY();
                        double d = Math.sqrt(dx * dx + dy * dy);

                        if (d < nearestDistance || nearestDistance == -1) {
                            nearestDistance = d;
                            nearestUnit = u2;
                        }
                    }
                }
            }
            return nearestUnit;
        }
    }
}
