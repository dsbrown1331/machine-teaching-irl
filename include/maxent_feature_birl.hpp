
#ifndef feature_birl_h
#define feature_birl_h

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <numeric>
#include <math.h>
#include "mdp.hpp"
#include "unit_norm_sampling.hpp"
#include "optimalTeaching.hpp"

using namespace std;

class FeatureBIRL { // BIRL process
      
   protected:
      
      double r_min, r_max, step_size;
      unsigned int chain_length;
      unsigned int grid_height, grid_width;
      double alpha;
      unsigned int iteration;
      int sampling_flag;
      bool mcmc_reject; //If false then it uses Yuchen's sample until accept method, if true uses normal MCMC sampling procedure
      int num_steps; //how many times to change current to get proposal
      int posterior_flag;
      double alpha_info;
      int horizon;
      double numTolerance;
      
      void initializeMDP();
      vector<pair<unsigned int,unsigned int> > positive_demonstration;
      vector<pair<unsigned int,unsigned int> > negative_demonstration;
      void modifyFeatureWeightsRandomly(FeatureGridMDP * gmdp, double step_size);
      void sampleL1UnitBallRandomly(FeatureGridMDP * gmdp);
      void updownL1UnitBallWalk(FeatureGridMDP * gmdp, double step);
      void manifoldL1UnitBallWalk(FeatureGridMDP * gmdp, double step, int num_steps);
      void manifoldL1UnitBallWalkAllSteps(FeatureGridMDP * gmdp, double step);


      double* posteriors = nullptr;
      FeatureGridMDP* MAPmdp = nullptr;
      double MAPposterior;
      
   public:
   
     FeatureGridMDP* mdp = nullptr; //original MDP 
     FeatureGridMDP** R_chain = nullptr; //storing the rewards along the way
      
      
     ~FeatureBIRL(){
        if(R_chain != nullptr) {
          for(unsigned int i=0; i<chain_length; i++) delete R_chain[i];
          delete []R_chain;
        }
        if(posteriors != nullptr) delete []posteriors;
        delete MAPmdp;
        
     }
      
     
     FeatureBIRL(FeatureGridMDP* init_mdp, double min_reward, double max_reward, unsigned int chain_len, double step, double conf, int samp_flag=0, bool reject=false, int num_step=1, int post_flag=0, double conf_2=1.0, int _horizon=-1, double numTol=0.00001 ):  
            r_min(min_reward), r_max(max_reward), step_size(step), chain_length(chain_len), alpha(conf), sampling_flag(samp_flag), mcmc_reject(reject), num_steps(num_step), posterior_flag(post_flag), alpha_info(conf_2), horizon(_horizon), numTolerance(numTol){ 
        unsigned int grid_height = init_mdp -> getGridHeight();
        unsigned int grid_width = init_mdp -> getGridWidth();
        bool* initStates = init_mdp -> getInitialStates();
        bool* termStates = init_mdp -> getTerminalStates();
        unsigned int nfeatures = init_mdp -> getNumFeatures();
        double* fweights = init_mdp -> getFeatureWeights();
        double** sfeatures = init_mdp -> getStateFeatures();
        bool stochastic = init_mdp -> isStochastic();
        double gamma = init_mdp -> getDiscount();
        //copy init_mdp
        mdp = new FeatureGridMDP(grid_width, grid_height, initStates, termStates, nfeatures, fweights, sfeatures, stochastic, gamma);
        mdp->setWallStates(init_mdp->getWallStates());
        initializeMDP(); //set weights to (r_min+r_max)/2
        
        MAPmdp = new FeatureGridMDP(grid_width, grid_height, initStates, termStates, nfeatures, fweights, sfeatures, stochastic, gamma);
        MAPmdp->setWallStates(init_mdp->getWallStates());
        MAPmdp->setFeatureWeights(mdp->getFeatureWeights());

        MAPposterior = 0;
        
        R_chain = new FeatureGridMDP*[chain_length];
        posteriors = new double[chain_length];    
        iteration = 0;
        
       }; 
       
      FeatureGridMDP* getMAPmdp(){return MAPmdp;}
      double getMAPposterior(){return MAPposterior;}
      void addPositiveDemo(pair<unsigned int,unsigned int> demo) { positive_demonstration.push_back(demo); }; // (state,action) pair
      void addNegativeDemo(pair<unsigned int,unsigned int> demo) { negative_demonstration.push_back(demo); };
      void addPositiveDemos(vector<pair<unsigned int,unsigned int> > demos);
      void addNegativeDemos(vector<pair<unsigned int,unsigned int> > demos);
      void run(double eps=0.001);
      void displayPositiveDemos();
      void displayNegativeDemos();
      void displayDemos();
      double getMinReward(){return r_min;};
      double getMaxReward(){return r_max;};
      double getStepSize(){return step_size;};
      unsigned int getChainLength(){return chain_length;};
      vector<pair<unsigned int,unsigned int> >& getPositiveDemos(){ return positive_demonstration; };
      vector<pair<unsigned int,unsigned int> >& getNegativeDemos(){ return negative_demonstration; };
      FeatureGridMDP** getRewardChain(){ return R_chain; };
      FeatureGridMDP* getMeanMDP(int burn, int skip);
      double* getPosteriorChain(){ return posteriors; };
      FeatureGridMDP* getMDP(){ return mdp;};
      double calculatePosterior(FeatureGridMDP* gmdp);
      double calculateMaxEntPosterior(FeatureGridMDP* gmdp);
      double calculateOptTeachingPosterior(FeatureGridMDP *gmdp);
      double calculateBIOPosterior(FeatureGridMDP *gmdp);
      double logsumexp(double* nums, unsigned int size);
      bool isDemonstration(pair<double,double> s_a);
      double getNumTolerance(){ return numTolerance; };
           
};

void FeatureBIRL::run(double eps)
{
   cout << "running BIRL" << endl;
     //cout.precision(10);
    //cout << "itr: " << iteration << endl;
    //clear out previous values if they exist
    if(iteration > 0) for(unsigned int i=0; i<chain_length-1; i++) delete R_chain[i];
    iteration++;
    MAPposterior = 0;
    R_chain[0] = mdp; // so that it can be deleted with R_chain!!!!
    //vector<unsigned int> policy (mdp->getNumStates());

    mdp->valueIteration(eps);//deterministicPolicyIteration(policy);
    mdp->calculateQValues();
    //mdp->displayFeatureWeights();
    double posterior = 0; //default
    if(posterior_flag == 0)
        posterior = calculatePosterior(mdp);
    else if(posterior_flag == 1)
        posterior = calculateMaxEntPosterior(mdp);
    else if(posterior_flag == 2)
        posterior = calculateOptTeachingPosterior(mdp);
    else if(posterior_flag == 3)
        posterior = calculateBIOPosterior(mdp);
    //cout << "init posterior: " << posterior << endl;
    posteriors[0] = exp(posterior); 
    int reject_cnt = 0;
    //BIRL iterations 
    for(unsigned int itr=1; itr < chain_length; itr++)
    {
      if(itr % 100 == 0)
         cout << "itr: " << itr << endl;
      FeatureGridMDP* temp_mdp = new FeatureGridMDP (mdp->getGridWidth(),mdp->getGridHeight(), mdp->getInitialStates(), mdp->getTerminalStates(), mdp->getNumFeatures(), mdp->getFeatureWeights(), mdp->getStateFeatures(), mdp->isStochastic(), mdp->getDiscount());
      temp_mdp->setWallStates(mdp->getWallStates());
      
      temp_mdp->setFeatureWeights(mdp->getFeatureWeights());
      if(sampling_flag == 0)
      {   //random grid walk
          modifyFeatureWeightsRandomly(temp_mdp,step_size);
      }
      else if(sampling_flag == 1)
      {
          //cout << "sampling randomly from L1 unit ball" << endl;
          sampleL1UnitBallRandomly(temp_mdp);  
      }
      //updown sampling on L1 ball
      else if(sampling_flag == 2)
      { 
          //cout << "before step" << endl;
          //temp_mdp->displayFeatureWeights();
          updownL1UnitBallWalk(temp_mdp, step_size);
          //cout << "after step" << endl;
          //temp_mdp->displayFeatureWeights();
          //check if norm is right
          assert(isEqual(l1_norm(temp_mdp->getFeatureWeights(), temp_mdp->getNumFeatures()),1.0));
      }
      //random manifold walk sampling
      else if(sampling_flag == 3)
      {
          manifoldL1UnitBallWalk(temp_mdp, step_size, num_steps);
          assert(isEqual(l1_norm(temp_mdp->getFeatureWeights(), temp_mdp->getNumFeatures()),1.0));
      }
      else if(sampling_flag == 4)
      {
          manifoldL1UnitBallWalkAllSteps(temp_mdp, step_size);
          assert(isEqual(l1_norm(temp_mdp->getFeatureWeights(), temp_mdp->getNumFeatures()),1.0));
      }
      //cout << "trying out" << endl;    
      //temp_mdp->displayFeatureWeights();
      
      temp_mdp->valueIteration(eps, mdp->getValues());
      //temp_mdp->deterministicPolicyIteration(policy);//valueIteration(0.05);
      temp_mdp->calculateQValues();
      
      double new_posterior = 0;
      if(posterior_flag == 0)
          new_posterior = calculatePosterior(temp_mdp);
      else if(posterior_flag == 1)
          new_posterior = calculateMaxEntPosterior(temp_mdp);
      else if(posterior_flag == 2)
          new_posterior = calculateOptTeachingPosterior(temp_mdp);
      else if(posterior_flag == 3)
          new_posterior = calculateBIOPosterior(temp_mdp);
      
      //cout << "posterior: " << new_posterior << endl;
      double probability = min((double)1.0, exp(new_posterior - posterior));
      //cout << probability << endl;

      //transition with probability
      double r = ((double) rand() / (RAND_MAX));
      if ( r < probability ) //policy_changed && 
      {
         //temp_mdp->displayFeatureWeights();
         //cout << "accept" << endl;
         mdp = temp_mdp;
         posterior = new_posterior;
         R_chain[itr] = temp_mdp;
         posteriors[itr] = exp(new_posterior);
         //if (itr%100 == 0) cout << itr << ": " << posteriors[itr] << endl;
         if(posteriors[itr] > MAPposterior)
         {
           MAPposterior = posteriors[itr];
           //TODO remove set terminals, right? why here in first place?
           MAPmdp->setFeatureWeights(mdp->getFeatureWeights());
         }
      }else {
        //delete temp_mdp
        delete temp_mdp;
         //keep previous reward in chain
         //cout << "reject!!!!" << endl;
         reject_cnt++;
         if(mcmc_reject)
         {
            //TODO can I make this more efficient by adding a count variable?
            //make a copy of mdp
            FeatureGridMDP* mdp_copy = new FeatureGridMDP (mdp->getGridWidth(),mdp->getGridHeight(), mdp->getInitialStates(), mdp->getTerminalStates(), mdp->getNumFeatures(), mdp->getFeatureWeights(), mdp->getStateFeatures(), mdp->isStochastic(), mdp->getDiscount());
            mdp_copy->setValues(mdp->getValues());
            mdp_copy->setQValues(mdp->getQValues());
            mdp_copy->setWallStates(mdp->getWallStates());
            R_chain[itr] = mdp_copy;
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

//use counter factuals and truncate likelihood if demo isn't optimal under reward
double FeatureBIRL::calculateBIOPosterior(FeatureGridMDP* gmdp) //assuming uniform prior
{
  vector<pair<unsigned int, unsigned int> > demo = positive_demonstration;


  //first do a softmax to see if likely given reward
  //TODO: do I need this?
  //gmdp->valueIteration(numTolerance);
  //gmdp->calculateQValues();
  double log_softmax = calculatePosterior(gmdp);
  //cout << "softmax = " <<   exp(log_softmax) << endl;
  
  

  double likelihood = 0;
  
  //TODO: how to set K and horizon?
  int K = 1;
  bool stochasticPolicy = false;
  //int horizon = 3; //assume single state action pairs are chosen
  int numDemos = demo.size() / horizon;  //TODO: hack for now


  //try soft set cover
  
  //find Feasible(R)
  vector<vector<double> > feasibleR = getFeasibleRegion(gmdp);
  /*** Debug ***
  cout << "----Feasible(R) half spaces" << endl;
  for(vector<double> constr : feasibleR)
  {
     for(double c : constr)
         cout << c << ",";
     cout << endl;

  }
  *** End Debug ***/

  //find Feasible(D|R)
  map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts;
  vector< vector<double> > opt_policy = gmdp->getOptimalStochasticPolicy();
  vector<unsigned int> det_pi (gmdp->getNumStates());
  gmdp->getOptimalPolicy(det_pi);
  if(stochasticPolicy)
    sa_fcounts = calculateStateActionFCounts(opt_policy, gmdp, numTolerance); 
  else
    sa_fcounts = calculateStateActionFCounts(det_pi, gmdp, numTolerance);

  vector<vector<double> > feasibleDemo = getDemonstrationFeasibleRegion(demo, sa_fcounts, gmdp);
   /*** Debug ***
  cout << "----Feasible(D|R) half spaces" << endl;
  for(vector<double> constr : feasibleDemo)
  {
     for(double c : constr)
         cout << c << ",";
     cout << endl;

  }
   *** End Debug ***/

  //TODO: kind of a hack!
  // //check if |Feasible(D|R)| > |Feasible(R)| as heuristic to know if really bad match
  // if(feasibleDemo.size() > feasibleR.size())
  // {
  //   cout << "likelihood = " << 0.001 << endl;
  //   return log(0.001);
  // }
  
  double cumSim = calculateCumulativeSimilarity(feasibleDemo, feasibleR);
  
  //normalize
  //cout << "cumSim = " << cumSim << endl;
  double softMatch_acutal = cumSim / feasibleR.size();
  //cout << "softmatch actual = " << softMatch_acutal << endl;

  //calculate the best match possible in one demonstration

  //cout << "*** optimal teaching ***" << endl;
  vector<vector<pair<unsigned int, unsigned int > > > optTeachingSoln =solveSetCoverOptimalTeaching(gmdp, K, horizon, numTolerance, stochasticPolicy);
  //printOutDemos(optTeachingSoln);
  //TODO: maybe look at best demo rather than first...?
  //take first numDemos
  vector<pair<unsigned int, unsigned int> > opt_demo;
  unsigned int max_demos = 0;
  if(numDemos < optTeachingSoln.size())
    max_demos = numDemos;
  else
    max_demos = optTeachingSoln.size();
  //cout << optTeachingSoln.size() << endl;
  //cout << numDemos << endl;
  for(unsigned int d = 0; d < max_demos; d++)
    for(unsigned int h=0; h < horizon; h++)
      opt_demo.push_back(optTeachingSoln[d][h]); //TODO: hardcoded to assume all demonstrations are length horizon
  
  vector<vector<double> > feasibleOptDemo = getDemonstrationFeasibleRegion(opt_demo, sa_fcounts, gmdp);

  /*** Debug ***
  cout << "----Feasible Opt (D|R) half spaces" << endl;
  for(vector<double> constr : feasibleOptDemo)
  {
     for(double c : constr)
         cout << c << ",";
     cout << endl;

  }
   *** End Debug ***/
  double optCumSim = calculateCumulativeSimilarity(feasibleOptDemo, feasibleR);

  //normalize
  //cout << "opt cumSim = " << optCumSim << endl;
  double softMatch_optimal = optCumSim / feasibleR.size();
  //cout << "softmatch optimal = " << softMatch_optimal << endl;

  //likelihood = 1 - abs(softMatch_acutal - softMatch_optimal);
  likelihood = exp(-alpha_info * abs(softMatch_acutal - softMatch_optimal));

  //cout << "info likelihood = " << likelihood << endl;
  //cout << "total likelihood = " << likelihood * exp(log_softmax) << endl;
  return log(likelihood) + log_softmax;  
}
//TODO check that this follows guidance for softmax
double FeatureBIRL::logsumexp(double* nums, unsigned int size) {
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


//use optimal teaching to count number of halfspaces covered by demonstrations
double FeatureBIRL::calculateOptTeachingPosterior(FeatureGridMDP *gmdp)
{
    cout << "****************************" << endl;
    cout << "****************************" << endl;
    double eps = numTolerance;
    //int K = 1;
    //int trajLength = 1;
    double posterior = 0;
    double prior = 0;
    //TODO    
    posterior += prior;

    cout << "Hypothesis Features" << endl;
    gmdp->displayRewards();
    vector<unsigned int> opt_pi(gmdp->getNumStates());
    //cout << "sample opt policy" << endl;
    gmdp->getOptimalPolicy(opt_pi);
    gmdp->displayPolicy(opt_pi);
    //compute optimal teaching halfspace constraints for current mdp guess
    //output optimal teaching for debugging
    ////begin DEBUG
    cout << "====================" << endl;
//    cout << "optimal teaching deterministic" << endl;
//    vector<vector<pair<unsigned int, unsigned int> > > opt_demos = solveSetCoverOptimalTeaching(gmdp, K, trajLength);       
//     cout << "----- Solution ------" << endl;
//    int count = 0;
//    //print out demos
//    for(vector<pair<unsigned int, unsigned int> > traj : opt_demos)
//    {
//        cout << "demo " << count << endl;
//        count++;
//        for(pair<unsigned int, unsigned int> p : traj)
//            cout << "(" <<  p.first << "," << p.second << "), ";
//        cout << endl;
//    }    
    ////end Debug
    
    //calculate the halfspaces for gmdp
    vector<vector<double> > feasibleConstraints = getFeasibleRegion(gmdp);
    cout << " - Feasible(R) constraints - " << endl;
    for(vector<double> v : feasibleConstraints)
    {
      for(double dv : v)
        cout << dv << " ";
      cout << endl;
    }
    
    vector< vector<double> > opt_policy = gmdp->getOptimalStochasticPolicy();
    map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts = calculateStateActionFCounts(opt_policy, gmdp, eps); 
    
    //calculate the halfspaces for demos
    vector<vector<double> > demoConstraints = getAllConstraintsForTraj(positive_demonstration, sa_fcounts, gmdp);
    cout << " - Feasible(D|R) constraints - " << endl;
    for(vector<double> v : demoConstraints)
    {
      for(double dv : v)
        cout << dv << " ";
      cout << endl;
    }
    //count the overlap, number of faces of Feasible(R) covered by Feasible(D|R)
    int overlap = countCovered(demoConstraints, feasibleConstraints);
    cout << "covered = " << overlap << endl;
    
    //count how many halfspaces are covered by demonstrations
    double smoothing = .01; //add-one smoothing constant 
    double covered = overlap + smoothing;
   //cout << "opt count = " << optCount << endl;
   double normalizer = feasibleConstraints.size() + smoothing;
   //cout << "normalizer = " << normalizer << endl;
   cout << "posterior = " << covered / normalizer << endl;
   posterior = log( covered / normalizer );     
   cout << "log posterior = " << posterior << endl;
           
   //cout << "posterior" << posterior << endl;
   return posterior;

}

double FeatureBIRL::calculateMaxEntPosterior(FeatureGridMDP* gmdp) //assuming uniform prior
{
    
    double posterior = 0;
    //add in a zero norm (non-zero count)
    double prior = 0;
    //TODO    
    posterior += prior;
    unsigned int state, action;
    
    //count how many times the action taken by expert is in argmax of Q* for hypothesis reward R   
    double smoothing = 0; //smoothing constant 
    double optCount = smoothing;
    int demoCount = 0;
    // "-- Positive Demos --" 
   for(unsigned int i=0; i < positive_demonstration.size(); i++)
   {
      demoCount += 1;
      pair<unsigned int,unsigned int> demo = positive_demonstration[i];
      state =  demo.first;
      action = demo.second; 
      if(gmdp->isOptimalAction(state, action, 0.0001))
         optCount += 1;
        
   }
   //cout << "opt count = " << optCount << endl;
   double normalizer = demoCount + smoothing;
   //cout << "normalizer = " << normalizer << endl;
   posterior = alpha * log( optCount / normalizer );     
           
   //cout << "posterior" << posterior << endl;
   return posterior;
}



double FeatureBIRL::calculatePosterior(FeatureGridMDP* gmdp) //assuming uniform prior
{
    
    double posterior = 0;
    //add in a zero norm (non-zero count)
    double prior = 0;
//    double sigma = 1;
//    //laplacian prior over states
//    for(int i=0; i < gmdp->getNumStates(); i++)
//    {
//        double reward = gmdp->getReward(i);
//        prior -= abs(reward) / (2 * sigma);
//        
//    }
    
    posterior += prior;
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

void FeatureBIRL::modifyFeatureWeightsRandomly(FeatureGridMDP * gmdp, double step)
{
   unsigned int state = rand() % gmdp->getNumFeatures();
   double change = pow(-1,rand()%2)*step;
   //cout << "before " << gmdp->getReward(state) << endl;
   //cout << "change " << change << endl;
   double weight = max(min(gmdp->getWeight(state) + change, r_max), r_min);
   //if(gmdp->isTerminalState(state)) reward = max(min(gmdp->getReward(state) + change, r_max), 0.0);
   //else reward = max(min(gmdp->getReward(state) + change, 0.0), r_min); 
   //cout << "after " << reward << endl;
   gmdp->setFeatureWeight(state, weight);
}

void FeatureBIRL::sampleL1UnitBallRandomly(FeatureGridMDP * gmdp)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = sample_unit_L1_norm(numFeatures);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}

void FeatureBIRL::updownL1UnitBallWalk(FeatureGridMDP * gmdp, double step)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = updown_l1_norm_walk(gmdp->getFeatureWeights(), numFeatures, step);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}


void FeatureBIRL::manifoldL1UnitBallWalk(FeatureGridMDP * gmdp, double step, int num_steps)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = random_manifold_l1_step(gmdp->getFeatureWeights(), numFeatures, step, num_steps);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}

void FeatureBIRL::manifoldL1UnitBallWalkAllSteps(FeatureGridMDP * gmdp, double step)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = take_all_manifold_l1_steps(gmdp->getFeatureWeights(), numFeatures, step);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}


void FeatureBIRL::addPositiveDemos(vector<pair<unsigned int,unsigned int> > demos)
{
    for(unsigned int i=0; i < demos.size(); i++)  positive_demonstration.push_back(demos[i]);
    //positive_demonstration.insert(positive_demonstration.end(), demos.begin(), demos.end());
}
void FeatureBIRL::addNegativeDemos(vector<pair<unsigned int,unsigned int> > demos)
{
    for(unsigned int i=0; i < demos.size(); i++)  negative_demonstration.push_back(demos[i]);
    //negative_demonstration.insert(negative_demonstration.end(), demos.begin(), demos.end());
}

void FeatureBIRL::displayDemos()
{
   displayPositiveDemos();
   displayNegativeDemos();
}
      
void FeatureBIRL::displayPositiveDemos()
{
   if(positive_demonstration.size() !=0 ) cout << "\n-- Positive Demos --" << endl;
   for(unsigned int i=0; i < positive_demonstration.size(); i++)
   {
      pair<unsigned int,unsigned int> demo = positive_demonstration[i];
      cout << " (" << demo.first << "," << demo.second << ") "; 
   
   }
   cout << endl;
}
void FeatureBIRL::displayNegativeDemos()
{
   if(negative_demonstration.size() != 0) cout << "\n-- Negative Demos --" << endl;
   for(unsigned int i=0; i < negative_demonstration.size(); i++)
   {
      pair<unsigned int,unsigned int> demo = negative_demonstration[i];
      cout << " (" << demo.first << "," << demo.second << ") "; 
   
   }
   cout << endl;
}
void FeatureBIRL::initializeMDP()
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
       unsigned int numDims = mdp->getNumFeatures();
       double* weights = new double[numDims];
       for(unsigned int s=0; s<numDims; s++)
           weights[s] = -1.0 / numDims;
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
       //weights[0] = 1.0;
       mdp->setFeatureWeights(weights);
       delete [] weights;
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

bool FeatureBIRL::isDemonstration(pair<double,double> s_a)
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
