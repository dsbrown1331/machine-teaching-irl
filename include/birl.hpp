
#ifndef birl_h
#define birl_h

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <math.h>
#include "mdp.hpp"

using namespace std;

class BIRL { // BIRL process
      
   protected:
      
      
      double r_min, r_max, step_size;
      unsigned int chain_length;
      unsigned int grid_height, grid_width;
      double alpha;                     //confidence parameter
      
      unsigned int iteration;
      
      void initializeMDP();
      vector<pair<unsigned int,unsigned int> > positive_demonstration;
      vector<pair<unsigned int,unsigned int> > negative_demonstration;
      void modifyRewardRandomly(GridMDP * gmdp, double step_size);

      double* posteriors = nullptr;
      GridMDP* MAPmdp = nullptr;
      double MAPposterior;
      
   public:
   
     GridMDP* mdp = nullptr; //original MDP 
     GridMDP** R_chain = nullptr; //storing the rewards along the way
      
     ~BIRL(){
        if(R_chain != nullptr) {
          for(unsigned int i=0; i<chain_length; i++) delete R_chain[i];
          delete []R_chain;
        }
        if(posteriors != nullptr) delete []posteriors;
        delete MAPmdp;
        
     }
      
     
     BIRL(unsigned int width, unsigned int height, vector<unsigned int> initStates, vector<unsigned int> termStates, double min_reward, double max_reward, unsigned int chain, double step, double conf):  r_min(min_reward), r_max(max_reward), step_size(step), chain_length(chain), grid_height(height), grid_width(width), alpha(conf) { 
     
        mdp = new GridMDP(grid_width,grid_height, initStates, termStates);
        initializeMDP(); 
        
        MAPmdp = new GridMDP(grid_width,grid_height, initStates, termStates);
        
        MAPmdp->setRewards(mdp->getRewards());
        MAPposterior = 0;
        
        R_chain = new GridMDP*[chain_length];
        posteriors = new double[chain_length];    
        iteration = 0;
        
       }; 
       
      GridMDP* getMAPmdp(){return MAPmdp;}
      double getMAPposterior(){return MAPposterior;}
      void addPositiveDemo(pair<unsigned int,unsigned int> demo) { positive_demonstration.push_back(demo); }; // (state,action) pair
      void addNegativeDemo(pair<unsigned int,unsigned int> demo) { negative_demonstration.push_back(demo); };
      void addPositiveDemos(vector<pair<unsigned int,unsigned int> > demos);
      void addNegativeDemos(vector<pair<unsigned int,unsigned int> > demos);
      void run();
      void displayPositiveDemos();
      void displayNegativeDemos();
      void displayDemos();
      double getMinReward(){return r_min;};
      double getMaxReward(){return r_max;};
      double getStepSize(){return step_size;};
      unsigned int getChainLength(){return chain_length;};
      vector<pair<unsigned int,unsigned int> >& getPositiveDemos(){ return positive_demonstration; };
      vector<pair<unsigned int,unsigned int> >& getNegativeDemos(){ return negative_demonstration; };
      FeatureGridMDP* getMeanMDP(int burn, int skip);
      GridMDP** getRewardChain(){ return R_chain; };
      double* getPosteriorChain(){ return posteriors; };
      GridMDP* getMDP(){ return mdp;};
      double calculatePosterior(GridMDP* gmdp);
      double logsumexp(double* nums, unsigned int size);
      bool isDemonstration(pair<double,double> s_a);
           
};

void BIRL::run()
{

     cout.precision(10);
    //cout << "itr: " << iteration << endl;
    //clear out previous values if they exist
    if(iteration > 0) for(unsigned int i=0; i<chain_length-1; i++) delete R_chain[i];
    iteration++;
    MAPposterior = 0;
    R_chain[0] = mdp; // so that it can be deleted with R_chain!!!!
    //vector<unsigned int> policy (mdp->getNumStates());

    mdp->valueIteration(0.001);//deterministicPolicyIteration(policy);
    mdp->calculateQValues();
    double posterior = calculatePosterior(mdp);
    posteriors[0] = exp(posterior); 
    
    //BIRL iterations 
    for(unsigned int itr=1; itr < chain_length; itr++)
    {
      // deepcopy ?
      GridMDP* temp_mdp = new GridMDP (mdp->getGridWidth(),mdp->getGridHeight(), mdp->getInitialStates(), mdp->getTerminalStates());
      
      temp_mdp->setRewards(mdp->getRewards());
      modifyRewardRandomly(temp_mdp,step_size);
      
      temp_mdp->valueIteration(0.001, mdp->getValues());
      //temp_mdp->deterministicPolicyIteration(policy);//valueIteration(0.05);
      temp_mdp->calculateQValues();
      
      double new_posterior = calculatePosterior(temp_mdp);
      //cout << "posterior: " << new_posterior << endl;
      double probability = min((double)1.0, exp(new_posterior - posterior));

      if ( (rand() % 100)/100.0 < probability ) //policy_changed && 
      {
         mdp = temp_mdp;
         posterior = new_posterior;
         R_chain[itr] = temp_mdp;
         posteriors[itr] = exp(new_posterior);
         //if (itr%100 == 0) cout << itr << ": " << posteriors[itr] << endl;
         if(posteriors[itr] > MAPposterior)
         {
           MAPposterior = posteriors[itr];
           MAPmdp->setRewards(mdp->getRewards());
         }
      }else {
         itr--;
         delete temp_mdp;
      }
      
    
    }
  
}

double BIRL::logsumexp(double* nums, unsigned int size) {
  double max_exp = nums[0];
  double sum = 0.0;
  unsigned int i;

  for (i = 1 ; i < size ; i++)
  {
    if (nums[i] > max_exp)
      max_exp = nums[i];
   }

  for (i = 0; i < size ; i++)
    sum += exp(nums[i] - max_exp);

  return log(sum) + max_exp;
}

double BIRL::calculatePosterior(GridMDP* gmdp) //assuming uniform prior
{
    
    double posterior = 0;
    unsigned int state, action;
    unsigned int numActions = gmdp->getNumActions();
    
    // "-- Positive Demos --" 
   for(unsigned int i=0; i < positive_demonstration.size(); i++)
   {
      pair<unsigned int,unsigned int> demo = positive_demonstration[i];
      state =  demo.first;
      action = demo.second; 
      
      double Z [numActions]; //
      for(unsigned int a = 0; a < numActions; a++) Z[a] = alpha*(gmdp->getQValue(state,a));
      posterior += alpha*(gmdp->getQValue(state,action)) - logsumexp(Z, numActions);
      //cout << state << "," << action << ": " << posterior << endl;
   }
   
   // "-- Negative Demos --" 
   for(unsigned int i=0; i < negative_demonstration.size(); i++)
   {
      pair<unsigned int,unsigned int> demo = negative_demonstration[i];
      state =  demo.first;
      action = demo.second;
      double Z [numActions]; //
      for(unsigned int a = 0; a < numActions; a++)  Z[a] = alpha*(gmdp->getQValue(state,a));
      
      unsigned int ct = 0;
      double Z2 [numActions - 1]; 
      for(unsigned int a = 0; a < numActions; a++) 
      {
         if(a != action) Z2[ct++] = alpha*(gmdp->getQValue(state,a));
      }
      
      posterior += logsumexp(Z2, numActions-1) - logsumexp(Z, numActions);
   }
   //cout << "posterior" << posterior << endl;
   return posterior;
}

void BIRL::modifyRewardRandomly(GridMDP * gmdp, double step)
{
   unsigned int state = rand() % gmdp->getNumStates();
   double change = pow(-1,rand()%2)*step;
   //cout << "before " << gmdp->getReward(state) << endl;
   //cout << "change " << change << endl;
   double reward = max(min(gmdp->getReward(state) + change, r_max), r_min);
   //if(gmdp->isTerminalState(state)) reward = max(min(gmdp->getReward(state) + change, r_max), 0.0);
   //else reward = max(min(gmdp->getReward(state) + change, 0.0), r_min); 
   //cout << "after " << reward << endl;
   gmdp->setReward(state,reward);
}

void BIRL::addPositiveDemos(vector<pair<unsigned int,unsigned int> > demos)
{
    for(unsigned int i=0; i < demos.size(); i++)  positive_demonstration.push_back(demos[i]);
    //positive_demonstration.insert(positive_demonstration.end(), demos.begin(), demos.end());
}
void BIRL::addNegativeDemos(vector<pair<unsigned int,unsigned int> > demos)
{
    for(unsigned int i=0; i < demos.size(); i++)  negative_demonstration.push_back(demos[i]);
    //negative_demonstration.insert(negative_demonstration.end(), demos.begin(), demos.end());
}

void BIRL::displayDemos()
{
   displayPositiveDemos();
   displayNegativeDemos();
}
      
void BIRL::displayPositiveDemos()
{
   if(positive_demonstration.size() !=0 ) cout << "\n-- Positive Demos --" << endl;
   for(unsigned int i=0; i < positive_demonstration.size(); i++)
   {
      pair<unsigned int,unsigned int> demo = positive_demonstration[i];
      cout << " (" << demo.first << "," << demo.second << ") "; 
   
   }
   cout << endl;
}
void BIRL::displayNegativeDemos()
{
   if(negative_demonstration.size() != 0) cout << "\n-- Negative Demos --" << endl;
   for(unsigned int i=0; i < negative_demonstration.size(); i++)
   {
      pair<unsigned int,unsigned int> demo = negative_demonstration[i];
      cout << " (" << demo.first << "," << demo.second << ") "; 
   
   }
   cout << endl;
}
void BIRL::initializeMDP()
{
   double* rewards = new double[mdp->getNumStates()];
   for(unsigned int s=0; s<mdp->getNumStates(); s++)
   {
      rewards[s] = (r_min+r_max)/2;
   }
   mdp->setRewards(rewards);
   delete []rewards;
   

}

bool BIRL::isDemonstration(pair<double,double> s_a)
{
     for(unsigned int i=0; i < positive_demonstration.size(); i++)
   {
      if(positive_demonstration[i].first == s_a.first && positive_demonstration[i].second == s_a.second) return true;
   }
   for(unsigned int i=0; i < negative_demonstration.size(); i++)
   {
      if(negative_demonstration[i].first == s_a.first && negative_demonstration[i].second == s_a.second) return true;
   }
   return false;

}

FeatureGridMDP* FeatureBIRL::getMeanMDP(int burn, int skip)
{
    //average rewards in chain
    int nFeatures = mdp->getNumFeatures();
    double aveWeights[nFeatures];
    
    for(int i=0;i<nFeatures;i++) aveWeights[i] = 0;
    
    int count = 0;
    for(unsigned int i=burn; i<chain_length; i+=skip)
    {
        count++;
        //(*(R_chain + i))->displayFeatureWeights();
        //cout << "weights" << endl;
        double* w = (*(R_chain + i))->getFeatureWeights();
        for(int f=0; f < nFeatures; f++)
            aveWeights[f] += w[f];
        
    }
    for(int f=0; f < nFeatures; f++)
        aveWeights[f] /= count;
  
//    //create new MDP with average weights as features
    FeatureGridMDP* mean_mdp = new FeatureGridMDP(MAPmdp->getGridWidth(),MAPmdp->getGridHeight(), MAPmdp->getInitialStates(), MAPmdp->getTerminalStates(), MAPmdp->getNumFeatures(), aveWeights, MAPmdp->getStateFeatures(), MAPmdp->isStochastic(), MAPmdp->getDiscount());
    mean_mdp->setWallStates(MAPmdp->getWallStates());
    
    
    return mean_mdp;
}


#endif
