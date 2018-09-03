#ifndef confidence_bounds_qlearning_hpp
#define confidence_bounds_qlearning_hpp
#include "mdp.hpp"
#include <math.h>
#include "driving_world_demo.hpp"
#include "q_learner_driving_demo.hpp"
#include <string>
#include <unordered_map>

//double evaluateExpectedReturn(const vector<unsigned int> & policy, 
//                    const MDP* evalMDP, double eps);
                    
double evaluateExpectedReturn(TabularQLearner* policy, DrivingWorld* world, int numRollouts, int rolloutLength, double gamma);
double evaluateExpectedReturn(TabularQLearner* policy, DrivingWorld* world, int numRollouts, int rolloutLength, double gamma, int start_seed);

vector<double> getFeatureCounts(vector<State> & trajectories, double gamma);

double calculateWorstCaseFeatureCountBound(vector<State> & trajectory, TabularQLearner* policy, DrivingWorld* world, int numRollouts, int rolloutLength, double gamma);

vector<State> generateRollout(TabularQLearner* policy, DrivingWorld* world, int rolloutLength);

double calculateExpectedValueDifference(TabularQLearner* sample_policy, TabularQLearner* map_policy, DrivingWorld* sample_world, int mc_numRollouts, int mc_rolloutLength, double gamma);

vector<double> getExpectedFeatureCounts(TabularQLearner* policy, DrivingWorld* world, int numRollouts, int rolloutLength, double gamma);
                    
//void policyValueIteration(const vector<unsigned int> & policy, 
//                    const MDP* evalMDP, double eps, double* V);
//                    
//double getExpectedReturn(const MDP* mdp);
//double getAverageValueFromStartStates(double* V, bool* init, unsigned int numStates);
//double** calculateStateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps);
//double* calculateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps);
//double* calculateEmpiricalExpectedFeatureCounts(vector<vector<pair<unsigned int,unsigned int> > > trajectories, FeatureGridMDP* fmdp);

//double evaluateExpectedReturn(const vector<unsigned int> & policy, 
//                    const MDP* evalMDP, double eps)
//{
//    //initialize values to zero
//    unsigned int numStates = evalMDP->getNumStates();
//    double V[numStates];
//    for(unsigned int i=0; i<numStates; i++) V[i] = 0.0;
//    
//    //get value of policy in evalMDP
//    policyValueIteration(policy, evalMDP, eps, V);
//    
//    //get expected value of policy evaluated on evalMDP over starting dist
//    //TODO assumes uniform distibution over start states
//    bool* init = evalMDP->getInitialStates();
//    return getAverageValueFromStartStates(V, init, numStates);
//}

vector<double> getExpectedFeatureCounts(TabularQLearner* policy, DrivingWorld* world, int numRollouts, int rolloutLength, double gamma)
{
    vector<double> ave_policy_fcount(world->getNumRewardFeatures());
    //get estimate of policy feature counts
    for(int n = 0; n < numRollouts; n++)
    {
        vector<State> rollout = generateRollout(policy, world, rolloutLength);
        vector<double> traj_fcount = getFeatureCounts(rollout, gamma);
        //cumulative sum of feature counts
        for(unsigned int i=0; i<traj_fcount.size(); i++)
            ave_policy_fcount[i] += traj_fcount[i];
                
    
    }
    //compute average
    for(unsigned int f = 0; f < ave_policy_fcount.size(); f++)
        ave_policy_fcount[f] = ave_policy_fcount[f] / numRollouts;
    return ave_policy_fcount;
}

double calculateExpectedValueDifference(TabularQLearner* sample_policy, TabularQLearner* eval_policy, DrivingWorld* sample_world, int numRollouts, int rolloutLength, double gamma)
{
    //calculate expected feature counts for sample_policy and eval_policy
    vector<double> fcount_sample = getExpectedFeatureCounts(sample_policy, sample_world, numRollouts, rolloutLength, gamma);
    vector<double> fcount_eval = getExpectedFeatureCounts(eval_policy, sample_world, numRollouts, rolloutLength, gamma);
    double* fWeights = sample_world->getFeatureWeights();
    //calculate the dot product of world feature weights with fcount diff
    double evd = 0;
    for(unsigned int f = 0; f < sample_world->getNumRewardFeatures(); f++)
    {
        double diff = fcount_sample[f] - fcount_eval[f];
        evd += fWeights[f] * diff;
    }
    return evd;



}


vector<double> getFeatureCounts(vector<State> & trajectory, double gamma)
{
    vector<double> fcounts(trajectory[0].getRewardFeatures().size());
    int t = 0;
    for(State s : trajectory)
    {
        vector<int> features = s.getRewardFeatures();
        for(unsigned int i = 0; i < features.size(); i++)
        {
            fcounts[i] +=  pow(gamma, t) * features[i];
        }
        t++;
    }
    return fcounts;
        
}

//calculate based on trajectory and q-vals and take infintity norm of difference
double calculateWorstCaseFeatureCountBound(vector<State> & trajectory, TabularQLearner* policy, DrivingWorld* world, int numRollouts, int rolloutLength, double gamma)
{
    bool debug = true;
    //get estimate of expert feature counts
    vector<double> expert_fcount = getFeatureCounts(trajectory, gamma);

    
    vector<double> ave_policy_fcount(trajectory[0].getRewardFeatures().size());
    //get estimate of policy feature counts
    for(int n = 0; n < numRollouts; n++)
    {
        vector<State> rollout = generateRollout(policy, world, rolloutLength);
        vector<double> traj_fcount = getFeatureCounts(rollout, gamma);
        //cumulative sum of feature counts
        for(unsigned int i=0; i<traj_fcount.size(); i++)
            ave_policy_fcount[i] += traj_fcount[i];
                
    
    }
    //compute average
    for(unsigned int f = 0; f < ave_policy_fcount.size(); f++)
        ave_policy_fcount[f] = ave_policy_fcount[f] / numRollouts;

    if(debug)
    {
        cout << "demo fcounts" << endl;
        for(unsigned int f = 0; f < expert_fcount.size(); f++)
            cout << expert_fcount[f] << ",";
        cout << endl;
        cout << "policy fcounts" << endl;
        for(unsigned int f = 0; f < ave_policy_fcount.size(); f++)
            cout << ave_policy_fcount[f] << ", ";
        cout << endl;
    }
    
    //calculate the infinity norm of the difference
    double maxAbsDiff = 0;
    for(unsigned int f = 0; f < ave_policy_fcount.size(); f++)
    {
        double absDiff = abs(expert_fcount[f] - ave_policy_fcount[f]);
        if(absDiff > maxAbsDiff)
            maxAbsDiff = absDiff;
    }
    return maxAbsDiff;

}


vector<State> generateRollout(TabularQLearner* policy, DrivingWorld* world, int rolloutLength)
{
    vector<State> rollout;
    world->setVisuals(false);
    //eval_world.setVisuals(true); //also visualize the demonstration
    State initState = world->startNewEpoch();
    //rollout.push_back(initState);//only get rewards after taking a step!
    State state = initState;

   
    for(int t = 0; t < rolloutLength; t++)
    {
        //cout << "============" << step << endl;
        unsigned int action = policy->getArgmaxQvalues(state);
        //save state-action pair in demonstration
        //pair<string, unsigned int> sa_demo = make_pair(state.toString(), action);
//            if(remove_duplicates)
//            {
//                //check not in demo already
//                if (std::find(demonstration.begin(), demonstration.end(), sa_demo) == demonstration.end())
//                    demonstration.push_back(sa_demo);
//            }
//            else
//                demonstration.push_back(sa_demo);
        pair<State,double> nextStateReward = world->updateState(action);
        State nextState = nextStateReward.first;
        rollout.push_back(nextStateReward.first);
        state = nextState;
    }

   
    return rollout;


}


//compute expected return for driving world using monte carlo rollouts
double evaluateExpectedReturn(TabularQLearner* policy, DrivingWorld* world, int numRollouts, int rolloutLength, double gamma)
{
    world->setVisuals(false);
    //eval_world.setVisuals(true); //also visualize the demonstration

    double cum_reward = 0;
    for(int r = 0; r < numRollouts; r++)
    {
        State initState = world->startNewEpoch();
        State state = initState;
        double discounted_reward = 0;
        for(int t = 0; t < rolloutLength; t++)
        {
            //cout << "============" << step << endl;
            unsigned int action = policy->getArgmaxQvalues(state);
            //save state-action pair in demonstration
            //pair<string, unsigned int> sa_demo = make_pair(state.toString(), action);
//            if(remove_duplicates)
//            {
//                //check not in demo already
//                if (std::find(demonstration.begin(), demonstration.end(), sa_demo) == demonstration.end())
//                    demonstration.push_back(sa_demo);
//            }
//            else
//                demonstration.push_back(sa_demo);
            pair<State,double> nextStateReward = world->updateState(action);
            State nextState = nextStateReward.first;
            double reward = nextStateReward.second;
            discounted_reward += pow(gamma, t) * reward;
            state = nextState;
        }
        cum_reward += discounted_reward;
    }
    return cum_reward / numRollouts;
    
}

//compute expected return for driving world using monte carlo rollouts
double evaluateExpectedReturn(TabularQLearner* policy, DrivingWorld* world, int numRollouts, int rolloutLength, double gamma, int start_seed)
{
    world->setVisuals(false);
    //eval_world.setVisuals(true); //also visualize the demonstration

    double cum_reward = 0;
    for(int r = 0; r < numRollouts; r++)
    {
        State initState = world->startNewEpoch(start_seed);
        State state = initState;
        double discounted_reward = 0;
        for(int t = 0; t < rolloutLength; t++)
        {
            //cout << "============" << step << endl;
            unsigned int action = policy->getArgmaxQvalues(state);
            //save state-action pair in demonstration
            //pair<string, unsigned int> sa_demo = make_pair(state.toString(), action);
//            if(remove_duplicates)
//            {
//                //check not in demo already
//                if (std::find(demonstration.begin(), demonstration.end(), sa_demo) == demonstration.end())
//                    demonstration.push_back(sa_demo);
//            }
//            else
//                demonstration.push_back(sa_demo);
            pair<State,double> nextStateReward = world->updateState(action);
            State nextState = nextStateReward.first;
            double reward = nextStateReward.second;
            discounted_reward += pow(gamma, t) * reward;
            state = nextState;
        }
        cum_reward += discounted_reward;
    }
    return cum_reward / numRollouts;
    
}


//double getAverageValueFromStartStates(double* V, bool* init, unsigned int numStates)
//{
//    //check if there is at least one starting state
//    bool startStateExists = false;
//    for(unsigned int i=0; i<numStates; i++)
//        if(init[i])
//            startStateExists = true;
//    assert(startStateExists);
//    double valSum = 0;
//    int initCount = 0;
//    for(unsigned int s=0; s < numStates; s++)
//    {
//        if(init[s])
//        {
//            valSum += V[s];
//            initCount++;
//        }
//    }
//    return valSum / initCount;
//}

////Updates vector of values V to be value of using policy in evalMDP
////run value iteration until convergence using policy actions rather than argmax
//void policyValueIteration(const vector<unsigned int> & policy, 
//                    const MDP* evalMDP, double eps, double* V)
//{
//    double delta;
//    double discount = evalMDP->getDiscount();
//    double*** T = evalMDP->getTransitions();
//    //repeat until convergence within error eps
//    do
//    {
//        unsigned int numStates = evalMDP->getNumStates();

//        //cout << "--------" << endl;
//        //displayAsGrid(V);
//        delta = 0;
//        //update value of each state
//       // cout << eps * (1 - discount) / discount << "," << delta << endl;
//        
//        for(unsigned int s1 = 0; s1 < numStates; s1++)
//        {
//            double tempV = 0;
//            //add reward
//            tempV += evalMDP->getReward(s1);
//            //add discounted value of next state based on policy action
//            int policy_action = policy[s1];
//            //calculate expected utility of taking action a in state s1
//            double expUtil = 0;
//            
//            for(unsigned int s2 = 0; s2 < numStates; s2++)
//            {
//                expUtil += T[s1][policy_action][s2] * V[s2];
//            }
//            tempV += discount * expUtil;

//            //update delta to track convergence
//            double absDiff = abs(tempV - V[s1]);
//            if(absDiff > delta)
//                delta = absDiff;
//            V[s1] = tempV;
//        }
//        
//    }
//    while(delta > eps * (1 - discount) / discount);

//}

////returns the expected return of the optimal policy for the input mdp
////assumes value iteration has already been run
//double getExpectedReturn(const MDP* mdp)
//{
//    unsigned int numStates = mdp->getNumStates();
//    double* V = mdp->getValues();
//    bool* init = mdp->getInitialStates();
//    return getAverageValueFromStartStates(V, init, numStates);

//}

////uses an analogue to policy evaluation to calculate the expected features for each state
////runs until change is less than eps
//double** calculateStateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps)
//{
//    unsigned int numStates = fmdp->getNumStates();
//    unsigned int numFeatures = fmdp->getNumFeatures();
//    double** stateFeatures = fmdp->getStateFeatures();
//    double discount = fmdp->getDiscount();
//    double*** T = fmdp->getTransitions();


//    //initalize 2-d array for storing feature weights
//    double** featureCounts = new double*[numStates];
//    for(unsigned int s = 0; s < numStates; s++)
//        featureCounts[s] = new double[numFeatures];
//    for(unsigned int s = 0; s < numStates; s++)
//        for(unsigned int f = 0; f < numFeatures; f++)
//            featureCounts[s][f] = 0;

//    //run feature count iteration
//    double delta;
//    
//    //repeat until convergence within error eps
//    do
//    {
//            //////Debug
////           for(unsigned int s = 0; s < numStates; s++)
////            {
////                double* fcount = featureCounts[s];
////                cout << "State " << s << ": ";
////                for(unsigned int f = 0; f < numFeatures; f++)
////                    cout << fcount[f] << "\t";
////                cout << endl;
////            }    
////            cout << "-----------" << endl;
//            //////////

//        //cout << "--------" << endl;
//        //displayAsGrid(V);
//        delta = 0;
//        //update value of each state
//       // cout << eps * (1 - discount) / discount << "," << delta << endl;
//        
//        for(unsigned int s1 = 0; s1 < numStates; s1++)
//        {
//            //cout << "for state: " << s1 << endl;
//            //use temp array to store accumulated, discounted feature counts
//            double tempF[numFeatures];
//            for(unsigned int f = 0; f < numFeatures; f++)
//                tempF[f] = 0;            
//            
//            //add current state features
//            for(unsigned int f =0; f < numFeatures; f++)
//                tempF[f] += stateFeatures[s1][f];

//            //update value of each reachable next state following policy
//            unsigned int policyAction = policy[s1];
//            double transitionFeatures[numFeatures];
//            for(unsigned int f = 0; f < numFeatures; f++)
//                transitionFeatures[f] = 0;

//            for(unsigned int s2 = 0; s2 < numStates; s2++)
//            {
//                if(T[s1][policyAction][s2] > 0)
//                {       
//                    //cout << "adding transition to state: " << s2 << endl;
//                    //accumulate features for state s2
//                    for(unsigned int f = 0; f < numFeatures; f++)
//                        transitionFeatures[f] += T[s1][policyAction][s2] * featureCounts[s2][f];
//                }
//            }
//            //add discounted transition features to tempF
//            for(unsigned int f = 0; f < numFeatures; f++)
//            {
//                tempF[f] += discount * transitionFeatures[f];
//                //update delta to track convergence
//                double absDiff = abs(tempF[f] - featureCounts[s1][f]);
//                if(absDiff > delta)
//                    delta = absDiff;
//                featureCounts[s1][f] = tempF[f];
//            }
//        }
//        //cout << "delta " << delta << endl;
//    }
//    while(delta > eps);

//    return  featureCounts;
//}

//double* calculateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps)
//{
//    //average over initial state distribution (assumes all initial states equally likely)
//    double** stateFcounts = calculateStateExpectedFeatureCounts(policy, fmdp, eps);
//    unsigned int numStates = fmdp -> getNumStates();
//    unsigned int numFeatures = fmdp -> getNumFeatures();
//    int numInitialStates = 0;
//    
//    double* expFeatureCounts = new double[numFeatures];
//    fill(expFeatureCounts, expFeatureCounts + numFeatures, 0);
//    
//    for(unsigned int s = 0; s < numStates; s++)
//        if(fmdp -> isInitialState(s))
//        {
//            numInitialStates++;
//            for(unsigned int f = 0; f < numFeatures; f++)
//                expFeatureCounts[f] += stateFcounts[s][f];
//        }
//                
//    //divide by number of initial states
//    for(unsigned int f = 0; f < numFeatures; f++)
//        expFeatureCounts[f] /= numInitialStates;
//    
//    //clean up
//    for(unsigned int f = 0; f < numFeatures; f++)
//        delete[] stateFcounts[f];
//    delete stateFcounts;    
//    
//    return expFeatureCounts;

//}

//double* calculateEmpiricalExpectedFeatureCounts(vector<vector<pair<unsigned int,unsigned int> > > trajectories, FeatureGridMDP* fmdp)
//{
//    unsigned int numFeatures = fmdp->getNumFeatures();
//    double gamma = fmdp->getDiscount();
//    double** stateFeatures = fmdp->getStateFeatures();

//    //average over all trajectories the discounted feature weights
//    double* aveFeatureCounts = new double[numFeatures];
//    fill(aveFeatureCounts, aveFeatureCounts + numFeatures, 0);
//    for(vector<pair<unsigned int, unsigned int> > traj : trajectories)
//    {
//        for(unsigned int t = 0; t < traj.size(); t++)
//        {
//            pair<unsigned int, unsigned int> sa = traj[t];
//            unsigned int state = sa.first;
//            //cout << "adding features for state " << state << endl;
//            //for(unsigned int f = 0; f < numFeatures; f++)
//            //    cout << stateFeatures[state][f] << "\t";
//            //cout << endl;
//            for(unsigned int f = 0; f < numFeatures; f++)
//                aveFeatureCounts[f] += pow(gamma, t) * stateFeatures[state][f];
//        }
//    }
//    //divide by number of demos
//    for(unsigned int f = 0; f < numFeatures; f++)
//        aveFeatureCounts[f] /= trajectories.size();
//    return aveFeatureCounts;
//}




////calculate based on demos and policy and take infintity norm of difference
//double calculateWorstCaseFeatureCountBound(vector<unsigned int> & policy, FeatureGridMDP* fmdp, vector<vector<pair<unsigned int,unsigned int> > > trajectories, double eps)
//{
//    unsigned int numFeatures = fmdp -> getNumFeatures();
//    double* muhat_star = calculateEmpiricalExpectedFeatureCounts(trajectories,
//                                                                  fmdp);    
//    double* mu_pieval = calculateExpectedFeatureCounts(policy, fmdp, eps);
//    //calculate the infinity norm of the difference
//    double maxAbsDiff = 0;
//    for(unsigned int f = 0; f < numFeatures; f++)
//    {
//        double absDiff = abs(muhat_star[f] - mu_pieval[f]);
//        if(absDiff > maxAbsDiff)
//            maxAbsDiff = absDiff;
//    }
//    //clean up
//    delete[] muhat_star;
//    delete[] mu_pieval;     
//    return maxAbsDiff;
//}


#endif
