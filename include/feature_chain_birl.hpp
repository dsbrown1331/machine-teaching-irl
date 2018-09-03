
#ifndef feature_chain_birl_h
#define feature_chain_birl_h

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <list>
#include <numeric>
#include <math.h>
#include "mdp.hpp"
#include "../include/unit_norm_sampling.hpp"
#include "../include/optimalTeaching.hpp"

using namespace std;





class FeatureChainBIRL { // BIRL process
      
   protected:
      
      double r_min, r_max, step_size;
      unsigned int chain_length;
      double alpha;
      double info_alpha;
      unsigned int iteration;
      int sampling_flag;
      bool mcmc_reject; //If false then it uses Yuchen's sample until accept method, if true uses normal MCMC sampling procedure
      int num_steps; //how many times to change current to get proposal
      bool runBirl = false;
      int posterior_flag;
      
      
      void initializeMDP();
      vector<pair<unsigned int,unsigned int> > positive_demonstration;
      vector<pair<unsigned int,unsigned int> > negative_demonstration;
      void modifyFeatureWeightsRandomly(FeatureChainMDP * gmdp, double step_size);
      void sampleL1UnitBallRandomly(FeatureChainMDP * gmdp);
      void updownL1UnitBallWalk(FeatureChainMDP * gmdp, double step);
      void manifoldL1UnitBallWalk(FeatureChainMDP * gmdp, double step, int num_steps);
      void manifoldL1UnitBallWalkAllSteps(FeatureChainMDP * gmdp, double step);


      double* posteriors = nullptr;
      FeatureChainMDP* MAPmdp = nullptr;
      double MAPposterior;
      
   public:
   
     FeatureChainMDP* mdp = nullptr; //original MDP 
     FeatureChainMDP** R_chain = nullptr; //storing the rewards along the way
      
     ~FeatureChainBIRL(){
        if(R_chain != nullptr) {
          if(runBirl)
          {
            for(unsigned int i=0; i<chain_length; i++) 
              delete R_chain[i];
          }
          delete []R_chain;
        }
        if(posteriors != nullptr) delete []posteriors;
        if(MAPmdp != nullptr)
          delete MAPmdp;
        
     }
      
     double getAlpha(){return alpha;}
     
     FeatureChainBIRL(FeatureChainMDP* init_mdp, double min_reward, double max_reward, unsigned int chain_len, double step, double conf, double info_conf, int samp_flag=0, bool reject=false, int num_step=1, int post_flag=0):  
     r_min(min_reward), r_max(max_reward), step_size(step), chain_length(chain_len), alpha(conf), info_alpha(info_conf), sampling_flag(samp_flag), mcmc_reject(reject), num_steps(num_step), posterior_flag(post_flag){ 
     
        unsigned int numStates = init_mdp -> getNumStates();
        bool* initStates = init_mdp -> getInitialStates();
        bool* termStates = init_mdp -> getTerminalStates();
        unsigned int nfeatures = init_mdp -> getNumFeatures();
        double* fweights = init_mdp -> getFeatureWeights();
        double** sfeatures = init_mdp -> getStateFeatures();
        double gamma = init_mdp -> getDiscount();
        //copy init_mdp
        mdp = new FeatureChainMDP(numStates, initStates, termStates, nfeatures, fweights, sfeatures, gamma);
        initializeMDP(); //set weights to (r_min+r_max)/2
        
        
        MAPmdp = new FeatureChainMDP(numStates, initStates, termStates, nfeatures, fweights, sfeatures, gamma);
        
        MAPmdp->setFeatureWeights(mdp->getFeatureWeights());
        MAPposterior = 0;
        
        R_chain = new FeatureChainMDP*[chain_length];
        posteriors = new double[chain_length];    
        iteration = 0;
        
       }; 
       
      FeatureChainMDP* getMAPmdp(){return MAPmdp;}
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
      FeatureChainMDP** getRewardChain(){ return R_chain; };
      FeatureChainMDP* getMeanMDP(int burn, int skip);
      double* getPosteriorChain(){ return posteriors; };
      FeatureChainMDP* getMDP(){ return mdp;};
      double calculateBIOPosterior(FeatureChainMDP* gmdp);
      double logsumexp(double* nums, unsigned int size);
      double calculateLogSoftmax(FeatureChainMDP* gmdp);
      bool isDemonstration(pair<double,double> s_a);
           
};

void FeatureChainBIRL::run(double eps)
{
    runBirl = true;   
     //cout.precision(10);
    //cout << "itr: " << iteration << endl;
    //clear out previous values if they exist
    if(iteration > 0) for(unsigned int i=0; i<chain_length-1; i++) delete R_chain[i];
    iteration++;
    MAPposterior = 0;
    R_chain[0] = mdp; // so that it can be deleted with R_chain!!!!
    //vector<unsigned int> policy (mdp->getNumStates());
    //cout << "testing" << endl;
    mdp->valueIteration(eps);//deterministicPolicyIteration(policy);
    //cout << "value iter" << endl;
    mdp->calculateQValues();
    mdp->displayFeatureWeights();
    double posterior = 0;
    if(posterior_flag == 0)
      posterior = calculateLogSoftmax(mdp);
    else if(posterior_flag == 1)
      posterior = calculateBIOPosterior(mdp);
    //cout << "init posterior: " << posterior << endl;
    posteriors[0] = exp(posterior); 
    int reject_cnt = 0;
    //BIRL iterations 
    for(unsigned int itr=1; itr < chain_length; itr++)
    {
      //cout << "==============================" << endl;
      //cout << "itr: " << itr << endl;
      FeatureChainMDP* temp_mdp = new FeatureChainMDP (mdp->getNumStates(), mdp->getInitialStates(), mdp->getTerminalStates(), mdp->getNumFeatures(), mdp->getFeatureWeights(), mdp->getStateFeatures(), mdp->getDiscount());
      
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
      
      ////debub
      vector<unsigned int> eval_pi (temp_mdp->getNumStates());
      temp_mdp->calculateQValues();
      temp_mdp->getOptimalPolicy(eval_pi);
      //temp_mdp->displayPolicy(eval_pi);
      ////

      //temp_mdp->deterministicPolicyIteration(policy);//valueIteration(0.05);
      temp_mdp->calculateQValues();
      
      double new_posterior = 0;
      if(posterior_flag == 0) 
        new_posterior = calculateLogSoftmax(temp_mdp);
      else if(posterior_flag == 1)
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
            FeatureChainMDP* mdp_copy = new FeatureChainMDP (mdp->getNumStates(), mdp->getInitialStates(), mdp->getTerminalStates(), mdp->getNumFeatures(), mdp->getFeatureWeights(), mdp->getStateFeatures(), mdp->getDiscount());
            mdp_copy->setValues(mdp->getValues());
            mdp_copy->setQValues(mdp->getQValues());
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
//TODO check that this follows guidance for softmax
double FeatureChainBIRL::logsumexp(double* nums, unsigned int size) {
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

/*//try 1
double FeatureChainBIRL::calculatePosterior(FeatureChainMDP* gmdp) //assuming uniform prior
{
    cout << "demos" << endl;
    displayDemos();

    //try soft set cover
    double eps = 0.0001;
    
    //find Feasible(R)
    vector<vector<double> > feasibleR = getFeasibleRegion(gmdp);
    cout << "----Feasible(R) half spaces" << endl;
    for(vector<double> constr : feasibleR)
    {
       for(double c : constr)
           cout << c << ",";
       cout << endl;

    }

    //find Feasible(D|R)
    vector< vector<double> > opt_policy = gmdp->getOptimalStochasticPolicy();
    map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts = calculateStateActionFCounts(opt_policy, gmdp, eps); 

    vector<vector<double> > feasibleDemo = getAllConstraintsForTraj(positive_demonstration, sa_fcounts, gmdp);
    cout << "----Feasible(D|R) half spaces" << endl;
    for(vector<double> constr : feasibleDemo)
    {
       for(double c : constr)
           cout << c << ",";
       cout << endl;

    }

   

        //convert to linked list for easy removal and iteration
    list<vector<double> > feasibleRemaining;
    for(vector<double> constr : feasibleR)
        feasibleRemaining.push_back(constr);
    int sizeFeasibleR = feasibleR.size();

    double cumSim = 0;
    
    for(vector<double> constr : feasibleDemo)
    {
        // cout << "comparing : ";
        // for(double d : constr)
        //     cout << d << " ";
        // cout << endl;

        if(feasibleRemaining.size() == 0)
        {
           //cout << "no more matches possible" << endl;
            break;
        }
        
        list<vector<double> >::iterator iter, toRemove;
        vector<double> bestMatch;
        double bestSimilarity = -10;
        for(iter = feasibleRemaining.begin(); iter != feasibleRemaining.end(); iter++)
        {
            vector<double> match = *iter;
            // cout << "match possibility" << endl;
            // for(unsigned int i=0; i < match.size(); i++)
            //     cout << match[i] << ",";
            // cout << endl;
            
            double similarity = vectorDotProduct(constr, match);
            if(similarity > bestSimilarity)
            {
                bestSimilarity = similarity;
                bestMatch = match;
                toRemove = iter;
            }
        }
        // cout << "best match = ";
        // for(double d : bestMatch)
        //         cout << d << " ";
        // cout << endl;
        cumSim += bestSimilarity;
        
        //cout << "cum similarity = " << cumSim << endl;
        //remove best match
        feasibleRemaining.erase(toRemove);
        // cout << "feasibleRemaining" << endl;
        // for(vector<double> v : feasibleRemaining)
        // {
        //     for(double d : v)
        //         cout << d << " ";
        //     cout << endl;
        // } 
    }
    //transform to be in [0,1]
    cout << "cumSim = " << cumSim << endl;
    double likelihood = ((cumSim / sizeFeasibleR) + 1.0) / 2.0;
    cout << "likelihood = " << likelihood << endl;
    return log(likelihood);  

}*/

//try 2
// double FeatureChainBIRL::calculatePosterior(FeatureChainMDP* gmdp) //assuming uniform prior
// {
//     cout << "demos" << endl;
//     displayDemos();

//     //try soft set cover
//     double eps = 0.0001;
    
//     //find Feasible(R)
//     vector<vector<double> > feasibleR = getFeasibleRegion(gmdp);
//     cout << "----Feasible(R) half spaces" << endl;
//     for(vector<double> constr : feasibleR)
//     {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//     }

//     //find Feasible(D|R)
//     vector< vector<double> > opt_policy = gmdp->getOptimalStochasticPolicy();
//     map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts = calculateStateActionFCounts(opt_policy, gmdp, eps); 

//     vector<vector<double> > feasibleDemo = getAllConstraintsForTraj(positive_demonstration, sa_fcounts, gmdp);
//     cout << "----Feasible(D|R) half spaces" << endl;
//     for(vector<double> constr : feasibleDemo)
//     {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//     }

//     //check if |Feasible(D|R)| > |Feasible(R)| as heuristic to know if really bad match
//     if(feasibleDemo.size() > feasibleR.size())
//       return log(0.001);
    

//         //convert to linked list for easy removal and iteration
//     list<vector<double> > feasibleRemaining;
//     for(vector<double> constr : feasibleR)
//         feasibleRemaining.push_back(constr);
//     int sizeFeasibleR = feasibleR.size();

//     double cumSim = 0;
    
//     for(vector<double> constr : feasibleDemo)
//     {
//         // cout << "comparing : ";
//         // for(double d : constr)
//         //     cout << d << " ";
//         // cout << endl;

//         if(feasibleRemaining.size() == 0)
//         {
//            //cout << "no more matches possible" << endl;
//             break;
//         }
        
//         list<vector<double> >::iterator iter, toRemove;
//         vector<double> bestMatch;
//         double bestSimilarity = -10;
//         for(iter = feasibleRemaining.begin(); iter != feasibleRemaining.end(); iter++)
//         {
//             vector<double> match = *iter;
//             // cout << "match possibility" << endl;
//             // for(unsigned int i=0; i < match.size(); i++)
//             //     cout << match[i] << ",";
//             // cout << endl;
            
//             double similarity = vectorDotProduct(constr, match);
//             if(similarity > bestSimilarity)
//             {
//                 bestSimilarity = similarity;
//                 bestMatch = match;
//                 toRemove = iter;
//             }
//         }
//         // cout << "best match = ";
//         // for(double d : bestMatch)
//         //         cout << d << " ";
//         // cout << endl;
//         cumSim += bestSimilarity;
        
//         //cout << "cum similarity = " << cumSim << endl;
//         //remove best match
//         feasibleRemaining.erase(toRemove);
//         // cout << "feasibleRemaining" << endl;
//         // for(vector<double> v : feasibleRemaining)
//         // {
//         //     for(double d : v)
//         //         cout << d << " ";
//         //     cout << endl;
//         // } 
//     }
//     //transform to be in [0,1]
//     cout << "cumSim = " << cumSim << endl;
//     double likelihood = ((cumSim / sizeFeasibleR) + 1.0) / 2.0;
//     cout << "likelihood = " << likelihood << endl;
//     return log(likelihood);  

// }

//try 3
// double FeatureChainBIRL::calculatePosterior(FeatureChainMDP* gmdp) //assuming uniform prior
// {
//     cout << "demos" << endl;
//     displayDemos();

//     //try soft set cover
//     double eps = 0.0001;
    
//     //find Feasible(R)
//     vector<vector<double> > feasibleR = getFeasibleRegion(gmdp);
//     cout << "----Feasible(R) half spaces" << endl;
//     for(vector<double> constr : feasibleR)
//     {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//     }

//     //find Feasible(D|R)
//     vector< vector<double> > opt_policy = gmdp->getOptimalStochasticPolicy();
//     map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts = calculateStateActionFCounts(opt_policy, gmdp, eps); 

//     vector<vector<double> > feasibleDemo = getAllConstraintsForTraj(positive_demonstration, sa_fcounts, gmdp);
//     cout << "----Feasible(D|R) half spaces" << endl;
//     for(vector<double> constr : feasibleDemo)
//     {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//     }

//     //TODO: kind of a hack!
//     //check if |Feasible(D|R)| > |Feasible(R)| as heuristic to know if really bad match
//     if(feasibleDemo.size() > feasibleR.size())
//     {
//       cout << "likelihood = " << 0.001 << endl;
//       return log(0.001);
//     }
    

//         //convert to linked list for easy removal and iteration
    
//     list<vector<double> > feasibleRemaining;
//     for(vector<double> constr : feasibleR)
//         feasibleRemaining.push_back(constr);
//     int sizeFeasibleR = feasibleR.size();

//     double cumSim = 0;
    
//     for(vector<double> constr : feasibleDemo)
//     {
//         // cout << "comparing : ";
//         // for(double d : constr)
//         //     cout << d << " ";
//         // cout << endl;

//         if(feasibleRemaining.size() == 0)
//         {
//            //cout << "no more matches possible" << endl;
//             break;
//         }
        
//         list<vector<double> >::iterator iter, toRemove;
//         vector<double> bestMatch;
//         double bestSimilarity = -10;
//         for(iter = feasibleRemaining.begin(); iter != feasibleRemaining.end(); iter++)
//         {
//             vector<double> match = *iter;
//             // cout << "match possibility" << endl;
//             // for(unsigned int i=0; i < match.size(); i++)
//             //     cout << match[i] << ",";
//             // cout << endl;
            
//             double similarity = vectorDotProduct(constr, match);
//             if(similarity > bestSimilarity)
//             {
//                 bestSimilarity = similarity;
//                 bestMatch = match;
//                 toRemove = iter;
//             }
//         }
//         // cout << "best match = ";
//         // for(double d : bestMatch)
//         //         cout << d << " ";
//         // cout << endl;
//         cumSim += bestSimilarity;
        
//         //cout << "cum similarity = " << cumSim << endl;
//         //remove best match
//         feasibleRemaining.erase(toRemove);
//         // cout << "feasibleRemaining" << endl;
//         // for(vector<double> v : feasibleRemaining)
//         // {
//         //     for(double d : v)
//         //         cout << d << " ";
//         //     cout << endl;
//         // } 
//     }
//     //transform to be in [0,1]
//     cout << "cumSim = " << cumSim << endl;
//     double likelihood = ((cumSim / sizeFeasibleR) + 1.0) / 2.0;
//     cout << "likelihood = " << likelihood << endl;
//     return log(likelihood);  

// }

//use counter factuals and truncate likelihood if demo isn't optimal under reward
double FeatureChainBIRL::calculateBIOPosterior(FeatureChainMDP* gmdp) //assuming uniform prior
{
  vector<pair<unsigned int, unsigned int> > demo = positive_demonstration;
  double numTolerance = 0.00001;

  //first do a softmax to see if likely given reward
  //TODO: do I need this?
  //gmdp->valueIteration(numTolerance);
  //gmdp->calculateQValues();
  double log_softmax = calculateLogSoftmax(gmdp);
  cout << "softmax = " <<   exp(log_softmax) << endl;
  
  

  double likelihood = 0;
  
  //TODO: how to set K and horizon?
  int K = 10;
  int horizon = demo.size();


  //try soft set cover
  
  //find Feasible(R)
  vector<vector<double> > feasibleR = getFeasibleRegion(gmdp);
  /*** Debug ***
  // cout << "----Feasible(R) half spaces" << endl;
  // for(vector<double> constr : feasibleR)
  // {
  //    for(double c : constr)
  //        cout << c << ",";
  //    cout << endl;

  // }
  *** End Debug ***/

  //find Feasible(D|R)
  vector< vector<double> > opt_policy = gmdp->getOptimalStochasticPolicy();
  map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts = calculateStateActionFCounts(opt_policy, gmdp, numTolerance); 

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
  vector<vector<pair<unsigned int, unsigned int > > > optTeachingSoln =solveSetCoverOptimalTeaching(gmdp, K, horizon, numTolerance);
  //printOutDemos(optTeachingSoln);
  //TODO: maybe look at best demo rather than first...?
  //take first demonstration
  vector<pair<unsigned int, unsigned int> > opt_demo = optTeachingSoln[0];
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
  likelihood = exp(-info_alpha*abs(softMatch_acutal - softMatch_optimal));

  cout << "info likelihood = " << likelihood << endl;
  cout << "total likelihood = " << likelihood * exp(log_softmax) << endl;
  return log(likelihood) + log_softmax;  
}



//normal likelihood 
double FeatureChainBIRL::calculateLogSoftmax(FeatureChainMDP* gmdp) //assuming uniform prior
{
   
   double posterior = 0;
   //add in a zero norm (non-zero count)
   double prior = 0;
//    int count = 0;
//    double* weights = gmdp->getFeatureWeights();
//    for(int i=0; i < gmdp->getNumFeatures(); i++)
//        if(abs(weights[i]) > 0.0001)
//            count += 1;
//    prior = -1 * alpha * log(count-1);
   
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

void FeatureChainBIRL::modifyFeatureWeightsRandomly(FeatureChainMDP * gmdp, double step)
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

void FeatureChainBIRL::sampleL1UnitBallRandomly(FeatureChainMDP * gmdp)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = sample_unit_L1_norm(numFeatures);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}

void FeatureChainBIRL::updownL1UnitBallWalk(FeatureChainMDP * gmdp, double step)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = updown_l1_norm_walk(gmdp->getFeatureWeights(), numFeatures, step);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}


void FeatureChainBIRL::manifoldL1UnitBallWalk(FeatureChainMDP * gmdp, double step, int num_steps)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = random_manifold_l1_step(gmdp->getFeatureWeights(), numFeatures, step, num_steps);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}

void FeatureChainBIRL::manifoldL1UnitBallWalkAllSteps(FeatureChainMDP * gmdp, double step)
{
   unsigned int numFeatures = gmdp->getNumFeatures();
   double* newWeights = take_all_manifold_l1_steps(gmdp->getFeatureWeights(), numFeatures, step);
   gmdp->setFeatureWeights(newWeights);
   delete [] newWeights;
}


void FeatureChainBIRL::addPositiveDemos(vector<pair<unsigned int,unsigned int> > demos)
{
    for(unsigned int i=0; i < demos.size(); i++)  positive_demonstration.push_back(demos[i]);
    //positive_demonstration.insert(positive_demonstration.end(), demos.begin(), demos.end());
}
void FeatureChainBIRL::addNegativeDemos(vector<pair<unsigned int,unsigned int> > demos)
{
    for(unsigned int i=0; i < demos.size(); i++)  negative_demonstration.push_back(demos[i]);
    //negative_demonstration.insert(negative_demonstration.end(), demos.begin(), demos.end());
}

void FeatureChainBIRL::displayDemos()
{
   displayPositiveDemos();
   displayNegativeDemos();
}
      
void FeatureChainBIRL::displayPositiveDemos()
{
   if(positive_demonstration.size() !=0 ) cout << "\n-- Positive Demos --" << endl;
   for(unsigned int i=0; i < positive_demonstration.size(); i++)
   {
      pair<unsigned int,unsigned int> demo = positive_demonstration[i];
      cout << " (" << demo.first << "," << demo.second << ") "; 
   
   }
   cout << endl;
}
void FeatureChainBIRL::displayNegativeDemos()
{
   if(negative_demonstration.size() != 0) cout << "\n-- Negative Demos --" << endl;
   for(unsigned int i=0; i < negative_demonstration.size(); i++)
   {
      pair<unsigned int,unsigned int> demo = negative_demonstration[i];
      cout << " (" << demo.first << "," << demo.second << ") "; 
   
   }
   cout << endl;
}
void FeatureChainBIRL::initializeMDP()
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

bool FeatureChainBIRL::isDemonstration(pair<double,double> s_a)
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

FeatureChainMDP* FeatureChainBIRL::getMeanMDP(int burn, int skip)
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
    FeatureChainMDP* mean_mdp = new FeatureChainMDP(MAPmdp->getNumStates(), MAPmdp->getInitialStates(), MAPmdp->getTerminalStates(), MAPmdp->getNumFeatures(), aveWeights, MAPmdp->getStateFeatures(), MAPmdp->getDiscount());
   
    
    return mean_mdp;
}





#endif
