
#ifndef feature_birl_h
#define feature_birl_h

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <math.h>
#include "q_learner_driving.hpp"
#include "driving_world.hpp"
#include "unit_norm_sampling.hpp"

using namespace std;

class FeatureBIRL_Q { // BIRL process
      
   protected:
      DrivingWorld* world;
      unsigned int numFeatures;
      unsigned int numActions;
      int numSteps;
      double exploreRate;
      double learningRate;
      double gamma;
      double r_min;
      double r_max;
      unsigned int chain_length;
      double step_size;
      double alpha;
      unsigned int iteration;
      int sampling_flag;
      bool mcmc_reject; //If false then it uses Yuchen's sample until accept method, if true uses normal MCMC sampling procedure
      int num_steps; //how many times to change current to get proposal
      
      double* initializeWeights();
      vector<pair<string,unsigned int> > positive_demonstration;
      vector<pair<string,unsigned int> > negative_demonstration;
      void modifyFeatureWeightsRandomly(double* weights, double step_size);
      void sampleL1UnitBallRandomly(double* weights);
      void updownL1UnitBallWalk(double* weights, double step);
      void manifoldL1UnitBallWalk(double* weights, double step, int num_steps);
      void manifoldL1UnitBallWalkAllSteps(double* weights, double step);

      double* currentWeights = nullptr;
      double* posteriors = nullptr;
      double* MAPweights = nullptr;
      double MAPposterior;
      
   public:
   
     unordered_map<string,double>* qvals = nullptr; //current q-vals 
     double** R_chain = nullptr; //storing the feature weights along the way
      
     ~FeatureBIRL_Q(){
        if(R_chain != nullptr) {
          for(unsigned int i=0; i<chain_length; i++) delete R_chain[i];
          delete []R_chain;
        }
        if(posteriors != nullptr) delete []posteriors;
        
     }
      
     
     FeatureBIRL_Q(DrivingWorld* w, int numQsteps, double explrRate, double learnRate, double discount_factor, double min_reward, double max_reward, unsigned int chain_len, double step, double conf, int samp_flag=0, bool reject=false, int num_step=1): world(w), numSteps(numQsteps), exploreRate(explrRate), learningRate(learnRate), gamma(discount_factor),  r_min(min_reward), r_max(max_reward), chain_length(chain_len), step_size(step), alpha(conf), sampling_flag(samp_flag), mcmc_reject(reject), num_steps(num_step){ 
        numFeatures = world -> getNumRewardFeatures();
        numActions = world -> getNumActions();
        currentWeights = initializeWeights(); //set weights to (r_min+r_max)/2
        MAPweights = currentWeights;
        MAPposterior = 0;
        
        R_chain = new double*[chain_length];
        posteriors = new double[chain_length];    
        iteration = 0;
        
       }; 
       
      double* getMAPWeights(){return MAPweights;}
      double getMAPposterior(){return MAPposterior;}
      void addPositiveDemo(pair<string,unsigned int> demo) { positive_demonstration.push_back(demo); }; // (state,action) pair
      void addNegativeDemo(pair<string,unsigned int> demo) { negative_demonstration.push_back(demo); };
      void addPositiveDemos(vector<pair<string,unsigned int> > demos);
      void addNegativeDemos(vector<pair<string,unsigned int> > demos);
      void run();
      void displayPositiveDemos();
      void displayNegativeDemos();
      void displayDemos();
      double getMinReward(){return r_min;};
      double getMaxReward(){return r_max;};
      double getStepSize(){return step_size;};
      unsigned int getChainLength(){return chain_length;};
      vector<pair<string,unsigned int> >& getPositiveDemos(){ return positive_demonstration; };
      vector<pair<string,unsigned int> >& getNegativeDemos(){ return negative_demonstration; };
      double** getRewardChain(){ return R_chain; };
      double* getMeanWeights(int burn, int skip);
      double* getPosteriorChain(){ return posteriors; };
      double calculatePosterior(TabularQLearner* q);
      double logsumexp(double* nums, unsigned int size);
      bool isDemonstration(pair<string,unsigned int> s_a);
           
};

void FeatureBIRL_Q::run()
{
   
     //cout.precision(10);
    //cout << "itr: " << iteration << endl;
    //clear out previous values if they exist
    if(iteration > 0) for(unsigned int i=0; i<chain_length-1; i++) delete R_chain[i];
    iteration++;
    MAPposterior = 0;
    R_chain[0] = currentWeights; // so that it can be deleted with R_chain!!!!
    //vector<unsigned int> policy (mdp->getNumStates());
/////Solve using q-learning
    world->setFeatureWeights(currentWeights);
    TabularQLearner qbot(world, numActions, gamma);
    qbot.trainEpoch(numSteps, exploreRate, learningRate);
////
    double posterior = calculatePosterior(&qbot);
    //cout << "init posterior: " << posterior << endl;
    posteriors[0] = exp(posterior); 
    int reject_cnt = 0;
    //BIRL iterations 
    for(unsigned int itr=1; itr < chain_length; itr++)
    {
      if(itr % 50 == 0)
          cout << "itr: " << itr << endl;
      double* temp_weights = new double[numFeatures];
      //copy current weights to temp_weights
      for(unsigned int i=0;i<numFeatures;i++) temp_weights[i] = currentWeights[i];
      
      if(sampling_flag == 0)
      {   //random grid walk
          modifyFeatureWeightsRandomly(temp_weights,step_size);
      }
      else if(sampling_flag == 1)
      {
          //cout << "sampling randomly from L1 unit ball" << endl;
          sampleL1UnitBallRandomly(temp_weights);  
      }
      //updown sampling on L1 ball
      else if(sampling_flag == 2)
      { 
          //cout << "before step" << endl;
          //temp_mdp->displayFeatureWeights();
          updownL1UnitBallWalk(temp_weights, step_size);
          //cout << "after step" << endl;
          //temp_mdp->displayFeatureWeights();
          //check if norm is right
          assert(isEqual(l1_norm(temp_weights, numFeatures),1.0));
      }
      //random manifold walk sampling
      else if(sampling_flag == 3)
      {
          manifoldL1UnitBallWalk(temp_weights, step_size, num_steps);
          assert(isEqual(l1_norm(temp_weights, numFeatures),1.0));
      }
      else if(sampling_flag == 4)
      {
          manifoldL1UnitBallWalkAllSteps(temp_weights, step_size);
          assert(isEqual(l1_norm(temp_weights, numFeatures),1.0));
      }
      //debugging print out weights
      if(itr % 50 == 0)
      {
          cout << "weights" << endl;
          for(int i=0;i<numFeatures;i++) cout << temp_weights[i] << ",";
          cout << endl;
      }
      ////Solve world for q-values associated with temp_weights
      //set world weights to temp_weights
      world -> setFeatureWeights(temp_weights);
      TabularQLearner qbot(world, numActions, gamma);
      qbot.trainEpoch(numSteps, exploreRate, learningRate);
            
      double new_posterior = calculatePosterior(&qbot);
      //cout << "posterior: " << new_posterior << endl;
      double probability = min((double)1.0, exp(new_posterior - posterior));
      //cout << probability << endl;

      //transition with probability
      double r = ((double) rand() / (RAND_MAX));
      if ( r < probability ) //policy_changed && 
      {
         //temp_mdp->displayFeatureWeights();
         //cout << "accept" << endl;
         currentWeights = temp_weights;
         posterior = new_posterior;
         R_chain[itr] = temp_weights;
         posteriors[itr] = exp(new_posterior);
         //if (itr%100 == 0) cout << itr << ": " << posteriors[itr] << endl;
         if(posteriors[itr] > MAPposterior)
         {
           MAPposterior = posteriors[itr];
           //TODO remove set terminals, right? why here in first place?
           MAPweights = temp_weights;
         }
      }else {
        //delete temp_mdp
        delete temp_weights;
         //keep previous reward in chain
         //cout << "reject!!!!" << endl;
         reject_cnt++;
         if(mcmc_reject)
         {
            //TODO can I make this more efficient?
            //make a copy of mdp
            double* weights_copy = new double[numFeatures];
            for(unsigned int i=0;i<numFeatures;i++) 
                weights_copy[i] = currentWeights[i];
            R_chain[itr] = weights_copy;
         }
         //sample until you get accept and then add that -- doesn't repeat old reward in chain
         else
         {
             
             assert(reject_cnt < 100000);
             itr--;
             //delete temp_mdp;
         }
      }
      
    
    }
    cout << "rejects: " << reject_cnt << endl;
  
}
//TODO check that this follows guidance for softmax
double FeatureBIRL_Q::logsumexp(double* nums, unsigned int size) {
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

double FeatureBIRL_Q::calculatePosterior(TabularQLearner* qbot) //assuming uniform prior
{
    
    double posterior = 0;
    string state;
    unsigned int action;
    
    // "-- Positive Demos --" 
   for(unsigned int i=0; i < positive_demonstration.size(); i++)
   {
      pair<string,unsigned int> demo = positive_demonstration[i];
      state =  demo.first;
      action = demo.second; 
      
      double Z [numActions]; //
      //cout << "calculating posterior in birl" << endl;
      for(unsigned int a = 0; a < numActions; a++) Z[a] = alpha*(qbot->getQValue(state,a));
      posterior += alpha*(qbot->getQValue(state,action)) - logsumexp(Z, numActions);
      //cout << state << "," << action << ": " << posterior << endl;
   }
   
   // "-- Negative Demos --" 
   for(unsigned int i=0; i < negative_demonstration.size(); i++)
   {
      pair<string,unsigned int> demo = negative_demonstration[i];
      state =  demo.first;
      action = demo.second;
      double Z [numActions]; //
      for(unsigned int a = 0; a < numActions; a++)  Z[a] = alpha*(qbot->getQValue(state,a));
      
      unsigned int ct = 0;
      double Z2 [numActions - 1]; 
      for(unsigned int a = 0; a < numActions; a++) 
      {
         if(a != action) Z2[ct++] = alpha*(qbot->getQValue(state,a));
      }
      
      posterior += logsumexp(Z2, numActions-1) - logsumexp(Z, numActions);
   }
   //cout << "posterior" << posterior << endl;
   return posterior;
}

void FeatureBIRL_Q::modifyFeatureWeightsRandomly(double* weights, double step)
{
   unsigned int state = rand() % numFeatures;
   double change = pow(-1,rand()%2)*step;
   //cout << "before " << gmdp->getReward(state) << endl;
   //cout << "change " << change << endl;
   double weight = max(min(weights[state] + change, r_max), r_min);
   //if(gmdp->isTerminalState(state)) reward = max(min(gmdp->getReward(state) + change, r_max), 0.0);
   //else reward = max(min(gmdp->getReward(state) + change, 0.0), r_min); 
   //cout << "after " << reward << endl;
   weights[state] = weight;
}

void FeatureBIRL_Q::sampleL1UnitBallRandomly(double * weights)
{
   double* newWeights = sample_unit_L1_norm(numFeatures);
   for(unsigned int i=0; i<numFeatures;i++)
       weights[i] = newWeights[i];
   delete [] newWeights;
}

void FeatureBIRL_Q::updownL1UnitBallWalk(double* weights, double step)
{
   double* newWeights = updown_l1_norm_walk(weights, numFeatures, step);
   for(unsigned int i=0; i<numFeatures;i++)
       weights[i] = newWeights[i];
   delete [] newWeights;
}


void FeatureBIRL_Q::manifoldL1UnitBallWalk(double* weights, double step, int num_steps)
{
   double* newWeights = random_manifold_l1_step(weights, numFeatures, step, num_steps);
   for(unsigned int i=0; i<numFeatures;i++)
       weights[i] = newWeights[i];
   delete [] newWeights;
}

void FeatureBIRL_Q::manifoldL1UnitBallWalkAllSteps(double* weights, double step)
{
   double* newWeights = take_all_manifold_l1_steps(weights, numFeatures, step);
   for(unsigned int i=0; i<numFeatures;i++)
       weights[i] = newWeights[i];
   delete [] newWeights;
}


void FeatureBIRL_Q::addPositiveDemos(vector<pair<string,unsigned int> > demos)
{
    for(unsigned int i=0; i < demos.size(); i++)  positive_demonstration.push_back(demos[i]);
    //positive_demonstration.insert(positive_demonstration.end(), demos.begin(), demos.end());
}
void FeatureBIRL_Q::addNegativeDemos(vector<pair<string,unsigned int> > demos)
{
    for(unsigned int i=0; i < demos.size(); i++)  negative_demonstration.push_back(demos[i]);
    //negative_demonstration.insert(negative_demonstration.end(), demos.begin(), demos.end());
}

void FeatureBIRL_Q::displayDemos()
{
   displayPositiveDemos();
   displayNegativeDemos();
}
      
void FeatureBIRL_Q::displayPositiveDemos()
{
   if(positive_demonstration.size() !=0 ) cout << "\n-- Positive Demos --" << endl;
   for(unsigned int i=0; i < positive_demonstration.size(); i++)
   {
      pair<string,unsigned int> demo = positive_demonstration[i];
      cout << " (" << demo.first << "," << demo.second << ") "; 
   
   }
   cout << endl;
}
void FeatureBIRL_Q::displayNegativeDemos()
{
   if(negative_demonstration.size() != 0) cout << "\n-- Negative Demos --" << endl;
   for(unsigned int i=0; i < negative_demonstration.size(); i++)
   {
      pair<string,unsigned int> demo = negative_demonstration[i];
      cout << " (" << demo.first << "," << demo.second << ") "; 
   
   }
   cout << endl;
}
double* FeatureBIRL_Q::initializeWeights()
{
//   if(sampling_flag == 0)
//   {
//       double* weights = new double[mdp->getNumFeatures()];
//       for(unsigned int s=0; s<mdp->getNumFeatures(); s++)
//       {
//          weights[s] = (r_min+r_max)/2;
//       }
//       mdp->setFeatureWeights(weights);
//       delete [] weights;
//   }
//   else if (sampling_flag == 1)  //sample randomly from L1 unit ball
//   {
//       double* weights = sample_unit_L1_norm(mdp->getNumFeatures());
//       mdp->setFeatureWeights(weights);
//       delete [] weights;
//   }    
//   else if(sampling_flag == 2)
//   {
       unsigned int numDims = numFeatures;
       double* weights = new double[numDims];
       for(unsigned int s=0; s<numDims; s++)
           weights[s] = -1.0 / numFeatures;
//       {
//          if((rand() % 2) == 0)
//            weights[s] = 1.0 / numDims;
//          else
//            weights[s] = -1.0 / numDims;
////            if(s == 0)
////                weights[s] = 1.0;
////            else
////                weights[s] = 0.0;
//       }
//       weights[0] = 0.2;
//       weights[1] = 0.2;
//       weights[2] = -0.2;
//       weights[3] = 0.2;
//       weights[4] = 0.2;
//       weights[0] = 1.0;
       return weights;
//   }
//   else if(sampling_flag == 3)
//   {
//       unsigned int numDims = mdp->getNumFeatures();
//       double* weights = new double[numDims];
//       for(unsigned int s=0; s<numDims; s++)
//           weights[s] = 0.0;
////       {
////          if((rand() % 2) == 0)
////            weights[s] = 1.0 / numDims;
////          else
////            weights[s] = -1.0 / numDims;
//////            if(s == 0)
//////                weights[s] = 1.0;
//////            else
//////                weights[s] = 0.0;
////       }
////       weights[0] = 0.2;
////       weights[1] = 0.2;
////       weights[2] = -0.2;
////       weights[3] = 0.2;
////       weights[4] = 0.2;
//       weights[0] = 1.0;
//       mdp->setFeatureWeights(weights);
//       delete [] weights;
//   }

}

bool FeatureBIRL_Q::isDemonstration(pair<string, unsigned int> s_a)
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

double* FeatureBIRL_Q::getMeanWeights(int burn, int skip)
{
    //average rewards in chain
    double* aveWeights = new double[numFeatures];
    
    for(unsigned int i=0;i<numFeatures;i++) 
        aveWeights[i] = 0;
    
    int count = 0;
    for(unsigned int i=burn; i<chain_length; i+=skip)
    {
        count++;
        double* w = *(R_chain + i);
        for(unsigned int f=0; f < numFeatures; f++)
            aveWeights[f] += w[f];
        
    }
    for(unsigned int f=0; f < numFeatures; f++)
        aveWeights[f] /= count;
    
    return aveWeights;
}


#endif
