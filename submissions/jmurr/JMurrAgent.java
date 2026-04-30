/*
 * Template MCTS-based agent for MicroRTS LLM Competition.
 *
 * Instructions:
 * 1. Copy this folder to submissions/your-team-name/
 * 2. Rename this file to match your agent_class in metadata.json
 * 3. Update the package name below (replace "your_team_name" with your folder name, hyphens → underscores)
 * 4. Override methods to customize MCTS behavior
 * 5. Update metadata.json with your team info (set agent_file to this filename)
 */
package ai.mcts.submissions.jmurr;

import ai.RandomBiasedAI;
import ai.core.AI;
import ai.core.ParameterSpecification;
import ai.evaluation.EvaluationFunction;
import ai.evaluation.SimpleSqrtEvaluationFunction3;
import ai.mcts.naivemcts.NaiveMCTS;
import ai.mcts.naivemcts.NaiveMCTSNode;
import java.util.ArrayList;
import java.util.List;
import rts.GameState;
import rts.PlayerAction;
import rts.units.UnitTypeTable;

public class JMurrAgent extends NaiveMCTS {

    // Required constructor - must accept UnitTypeTable
    public JMurrAgent(UnitTypeTable utt) {
        // Parameters: time_budget, max_playouts, lookahead, max_depth,
        //             epsilon_l, epsilon_g, epsilon_0,
        //             playout_policy, evaluation_function, force_exploration
        super(100, -1, 100, 10,
              0.3f, 0.0f, 0.4f,
              new RandomBiasedAI(),
              new SimpleSqrtEvaluationFunction3(),
              true);
    }

    /*
     * Override getAction to customize behavior.
     * By default NaiveMCTS runs MCTS search and returns the best action.
     * You can add LLM calls here to bias the search or adjust parameters.
     */
    // @Override
    // public PlayerAction getAction(int player, GameState gs) throws Exception {
    //     // TODO: Add custom logic (e.g., LLM consultation, parameter tuning)
    //     return super.getAction(player, gs);
    // }

    public AI clone() {
        return new JMurrAgent(null);
    }

    @Override
    public List<ParameterSpecification> getParameters() {
        return new ArrayList<>();
    }

    @Override
    public boolean iteration(int player) throws Exception {
        
        NaiveMCTSNode leaf = tree.selectLeaf(player, 1-player, epsilon_l, epsilon_g, epsilon_0, global_strategy, MAX_TREE_DEPTH, current_iteration++);

        if (leaf!=null) {            
            GameState gs2 = leaf.gs.clone();
            simulate(gs2, gs2.getTime() + MAXSIMULATIONTIME);

            int time = gs2.getTime() - gs_to_start_from.getTime();
            double evaluation = ef.evaluate(player, 1-player, gs2)*Math.pow(0.99,time/10.0);
            double opp_evaluation = ef.evaluate(1-player, player, gs2)*Math.pow(0.99,time/10.0);
            double opp_factor = 0.25;
            opp_evaluation = opp_evaluation * opp_factor;

            leaf.propagateEvaluation(evaluation-opp_evaluation,null);            

            // update the epsilon values:
            epsilon_0*=discount_0;
            epsilon_l*=discount_l;
            epsilon_g*=discount_g;
            total_runs++;
            
//            System.out.println(total_runs + " - " + epsilon_0 + ", " + epsilon_l + ", " + epsilon_g);
            
        } else {
            // no actions to choose from :)
            System.err.println(this.getClass().getSimpleName() + ": claims there are no more leafs to explore...");
            return false;
        }
        return true;
    }
}
