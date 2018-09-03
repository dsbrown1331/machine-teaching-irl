
#ifndef q_learner_driving_h
#define q_learner_driving_h

#include <cstddef>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include <set>
#include <algorithm>
#include <iomanip>
#include <limits>
//#define NDEBUG         //uncomment this to disable all asserts
#include <assert.h> 
#include "driving_world_demo.hpp"
#include "mdp.hpp"


//TODO maybe write a wrapper for qvals that returns appropriately if not in map yet?

constexpr double min_double = std::numeric_limits<double>::lowest();


//    qbot.runEpoch(numSteps, exploreRate, learningRate);//    
//qbot.runEpoch(numSteps);
//    //test out learned policy
//    world.setVisuals(true)
//    State initState = world.startNewEpoch();
//    State state = initState;
//    cout << "init state: " << initState.toString() << endl;
//    for(int step = 0; step < 8; step++)
//    {
//        cout << "============" << step << endl;
//        int action = qbot.getArgMaxAction(state);


using namespace std;

class TabularQLearner { // General gridMDP q-learner
      
   protected:
      DrivingWorld* world;
      unsigned int numActions;
      double gamma;
      unordered_map<string, double> Qvals;
      bool isStateActionInQTable(string stateAction);
   public:
      TabularQLearner(DrivingWorld* w, double nActions, double discountFactor): world(w), numActions(nActions), gamma(discountFactor) {};
      
      
      ~TabularQLearner(){
      
      };
      double getQValue(State state, int action);
      double getQValue(string state, int action);
      void reset() { Qvals.clear();};
      
      
      unordered_map<string,double> getQValues() { 
        return Qvals;
      };

      set<string> getQTableStates();

      void trainEpoch(int maxSteps, double exploreProb, double learningRate);
      int getArgmaxQvalues(State state);
      int getArgmaxQvalues(string state);
      void displayQvalues();
      
};

set<string> TabularQLearner::getQTableStates()
{
    set<string> tableStates;
    unordered_map<string,double> qvals = getQValues();
    for(pair<string,double> p : qvals)
    {
        string state_action = p.first;
        size_t pos = state_action.find(":"); 
        string state = state_action.substr(0,pos);
        tableStates.insert(state);
    }
    return tableStates;

};

void TabularQLearner::displayQvalues()
{
    for(pair<string,double> p : Qvals)
    {
        cout << "State: " << p.first << endl;
        cout << "Qvalue: " << p.second << endl;
    }
}

bool TabularQLearner::isStateActionInQTable(string stateAction)
{
    unordered_map<string,double>::const_iterator contains = Qvals.find(stateAction);
    if ( contains == Qvals.end() )
        return false;
    else
        return true;
}

//if it returns -1 it means there are no Qvalues in hashtable
int TabularQLearner::getArgmaxQvalues(State state)
{
   double max_q = min_double;
   int max_action = -1;
   for(unsigned int a = 0; a < numActions; a++)
   {
      string stateAction = state.toStateString() + ":" + to_string(a);
      if( Qvals[stateAction] > max_q )
      {
         max_q  = Qvals[stateAction];
         max_action = a;
      }
 
   }
   //pick at random if there are ties
   vector<int> opt_actions;
   for(unsigned int a = 0; a < numActions; a++)
   {
      string stateAction = state.toStateString() + ":" + to_string(a);
      if( abs(Qvals[stateAction] - max_q) < 0.00001 )
        opt_actions.push_back(a);
   }     
   int rand_index = rand() % opt_actions.size();
   return opt_actions[rand_index];
}




//if it returns -1 it means there are no Qvalues in hashtable
int TabularQLearner::getArgmaxQvalues(string state)
{
   double max_q = min_double;
   int max_action = -1;
   //find max action
   for(unsigned int a = 0; a < numActions; a++)
   {
      string stateAction = state + ":" + to_string(a);
      if(isStateActionInQTable(stateAction))
      {
          if( Qvals[stateAction] > max_q )
          {
             max_q  = Qvals[stateAction];
             max_action = a;
          }
      }
   }
   return max_action;
}

//TODO not sure what to do if doesn't exist...
double TabularQLearner::getQValue(State state, int action)
{
    string stateAction = state.toStateString() + ":" + to_string(action);
    if(isStateActionInQTable(stateAction))
        return Qvals[stateAction];
    else
    {
        //cout << "State: " << state.toStateString() << ", action: " << action << endl;
        //cout << "!!!!!!!!!!Q-VALUE DOESNT EXIST!!!!!!!!!!!!" << endl;
        return -100;
    }

}

double TabularQLearner::getQValue(string state, int action)
{
    string stateAction = state + ":" + to_string(action);
    if(isStateActionInQTable(stateAction))
        return Qvals[stateAction];
    else
    {
        //cout << "State: " << state << ", action: " << action << endl;
        //cout << "!!!!!!!!!!Q-VALUE DOESNT EXIST!!!!!!!!!!!!" << endl;
        return -100;
    }

}

void TabularQLearner::trainEpoch(int numSteps, double exploreProb, double learningRate)
{
    int steps = 0;
    //get start state from world
    State state = world->startNewEpoch();
    while(steps < numSteps)
    {
        steps++;
        //pick action according to epsilon greedy policy
        double rand_value = rand() / (double) RAND_MAX;
        int action;
        if(rand_value < exploreProb)
        {
            //take random action
            action = rand() % numActions;
        }
        else //take greedy action
        {
            action = getArgmaxQvalues(state);
            //check if we should still roll dice if state never seen
            //if(action == -1)
            //    action = rand() % numActions;
        }
        //execute action in world and get result
        pair<State,double> nextStateReward = world->updateState(action);
        State nextState = nextStateReward.first;
        double reward = nextStateReward.second;
        //create hashable string by appending nextState.toStateString() with to_string(action)
        string stateAction = state.toStateString() + ":" + to_string(action);
        ////update Qvals
        //get next best action (random if nextState never seen before)
        int nextBestAction = getArgmaxQvalues(nextState);
        //if(nextBestAction == -1)
        //    nextBestAction = rand() % numActions;
        string nextStateAction = nextState.toStateString() + ":" + to_string(nextBestAction);
        Qvals[stateAction] = (1 - learningRate) * Qvals[stateAction]
                    + (learningRate * (reward + gamma * Qvals[nextStateAction]));
 
        //update state for next step
        state = nextState;
     }
    
}



#endif
