
#ifndef q_learner_grid_h
#define q_learner_grid_h

#include <cstddef>
#include <cmath>
#include <stdlib.h>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <limits>
//#define NDEBUG         //uncomment this to disable all asserts
#include <assert.h> 
#include "mdp.hpp"

constexpr double min_double = std::numeric_limits<double>::lowest();

#define NUM_ACTIONS 4

using namespace std;

class GridQLearner { // General gridMDP q-learner
      
   protected:
      GridMDP* world;
      double exploreProb;
      double learningRate;
      unsigned int numStates;
      unsigned int numActions;
      unsigned int gridHeight;
      unsigned int gridWidth;
      enum actions {UP, DOWN, LEFT, RIGHT};
      double** Qvals;
      bool** visited;
   public:
      GridQLearner(GridMDP* mdp): world(mdp){
        numStates = mdp->getNumStates();
        numActions = mdp->getNumActions();
        gridHeight = mdp->getGridHeight();
        gridWidth = mdp->getGridWidth();
        Qvals = new double*[numStates];
        for(unsigned int s = 0; s < numStates; s++) Qvals[s] = new double[numActions];
        for(unsigned int s = 0; s < numStates; s++)
            for(unsigned int a = 0; a < numActions; a++)
                Qvals[s][a] = 0.0;
      
        visited = new bool*[numStates];
        for(unsigned int s = 0; s < numStates; s++) visited[s] = new bool[numActions];
        for(unsigned int s = 0; s < numStates; s++)
            for(unsigned int a = 0; a < numActions; a++)
                visited[s][a] = false;
      
      };
      
      
      ~GridQLearner(){
      
          for(unsigned int s = 0; s < numStates; s++) 
              delete[] Qvals[s];
          delete[] Qvals;
      };
      double getQValue(unsigned int state,unsigned int action)
      { 
        return Qvals[state][action];
      };
      
      double** getQValues() { 
        return Qvals;
      };

      void trainEpoch(int maxSteps, double exploreProb, double learningRate);
      void getArgMaxPolicy(vector<unsigned int> & policy);
      void displayQValues();
      void displayPolicy(vector<unsigned int> & policy);
      unsigned int getArgmaxQvalues(unsigned int state);
      
};

unsigned int GridQLearner::getArgmaxQvalues(unsigned int state)
{
   double max_q = min_double;
   unsigned int max_action = 0;
   for(unsigned int a = 0; a < numActions; a++)
   {
      if( Qvals[state][a] > max_q )
      {
         max_q  = Qvals[state][a];
         max_action = a;
      }
   }
   return max_action;
}

void GridQLearner::getArgMaxPolicy(vector<unsigned int> & policy)
{
   for(unsigned int s = 0; s < numStates; s++)
    {
       unsigned int max_action = getArgmaxQvalues(s); 
       policy[s] = max_action;      
     }
}


void GridQLearner::trainEpoch(int numSteps, double exploreProb, double learningRate)
{
     int steps = 0;
     //pick start state at random from initial states
     vector<unsigned int> initStates;
     for(unsigned int s = 0; s < numStates; s++)
        if(world->isInitialState(s))
            initStates.push_back(s);
     int rand_indx = rand() % initStates.size();
     int state = initStates[rand_indx];
     int nextState = 0;
     //cout << "initial state: " << state << endl;
     while(steps < numSteps)
     {
         steps++;
         //get reward in current state
         double reward = world->getReward(state);
         //if state is terminal then update Qvals to terminal reward and end epoch
         if(world->isTerminalState(state))
         {
            for(unsigned int a = 0; a < numActions; a++)
                Qvals[state][a] = reward;
            break;
         }
         else
         {
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
             }
             //cout << "(state, action):" << "(" << state << "," << action << ")" << endl; 
             //feed action into MDP to get transition probs and pick next state
             vector<double> transitionProbs(numStates);
             for(unsigned int s2 = 0; s2 < numStates; s2++)
             {
                transitionProbs[s2] = world->getTransitions()[state][action][s2];
             }
             nextState = roulette_wheel(transitionProbs);

             //update Qvals
             int nextBestAction = getArgmaxQvalues(nextState);
             double gamma = world->getDiscount();
//             if(!visited[state][action])
//             {
//                Qvals[state][action] = reward;
//                visited[state][action] = true;
//             }
//             else
                Qvals[state][action] = (1 - learningRate) * Qvals[state][action]
                            + (learningRate  
                            * (reward + gamma * Qvals[nextState][nextBestAction]));
         }
         //update state for next step
         state = nextState;
     }
    
}


void GridQLearner::displayPolicy(vector<unsigned int> & policy)
{
    world->displayPolicy(policy);
}

void GridQLearner::displayQValues()
{
    ios::fmtflags f( cout.flags() );
    std::streamsize ss = std::cout.precision();
    if (Qvals == nullptr){
      cout << "ERROR: no Q values!" << endl;
      return;
    }
    cout << "-------- UP ----------" << endl;
    for(unsigned int r = 0; r < gridHeight; r++)
    {
        for(unsigned int c = 0; c < gridWidth; c++)
        {
            unsigned int state = r*gridWidth + c;
            cout << setiosflags(ios::fixed)
            << setprecision(3)
            << Qvals[state][UP] << "  ";
        }
        cout << endl;
    }
        cout << "-------- DOWN ----------" << endl;
    for(unsigned int r = 0; r < gridHeight; r++)
    {
        for(unsigned int c = 0; c < gridWidth; c++)
        {
            unsigned int state = r*gridWidth + c;
            cout << Qvals[state][DOWN] << "  ";
        }
        cout << endl;
    }

    cout << "-------- LEFT ----------" << endl;
    for(unsigned int r = 0; r < gridHeight; r++)
    {
        for(unsigned int c = 0; c < gridWidth; c++)
        {
            unsigned int state = r*gridWidth + c;
            cout << Qvals[state][LEFT] << "  ";
        }
        cout << endl;
    }

    cout << "-------- RIGHT ----------" << endl;
    for(unsigned int r = 0; r < gridHeight; r++)
    {
        for(unsigned int c = 0; c < gridWidth; c++)
        {
            unsigned int state = r*gridWidth + c;
            cout << Qvals[state][RIGHT] << "  ";
        }
        cout << endl;
    }
    cout.flags( f );
    cout << setprecision(ss);
}


#endif
