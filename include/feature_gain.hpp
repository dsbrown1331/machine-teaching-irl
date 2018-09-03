
#ifndef gain_h
#define gain_h

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <map>

#include <iostream>
#include <fstream>
#include <limits>

#include "mdp.hpp"


// Fill the zipped vector with pairs consisting of the
// corresponding elements of a and b. (This assumes 
// that the vectors have equal length)
using namespace std;

void zip(
    long double* a, 
    long double* b,
    unsigned int size,
    vector<pair<long double,long double> > &zipped)
{
  zipped.resize(size);
  for(unsigned int i=0; i<size; ++i)
  {
    zipped[i] = make_pair(a[i], b[i]);
  }
}

// Write the first and second element of the pairs in 
// the given zipped vector into a and b. (This assumes 
// that the vectors have equal length)

void unzip(
    const vector<pair<long double, long double> > &zipped, 
    long double* a, 
    long double* b)
{
  for(unsigned int i=0; i< zipped.size(); i++)
  {
    a[i] = zipped[i].first;
    b[i] = zipped[i].second;
  }
}

bool greaterThan(const pair<long double, long double>& a, const pair<long double, long double>& b)
{
  return a.first < b.first;
}

class FeatureInfoGainCalculator{

  private:
    FeatureBIRL * birl = nullptr;
    FeatureMDP* curr_mdp;
    unsigned int chain_length;


  public:
    FeatureInfoGainCalculator(FeatureBIRL* input_birl) {
      curr_mdp = input_birl->getMDP();
      chain_length = input_birl->getChainLength();
      birl = input_birl; 
    };
    
    ~ FeatureInfoGainCalculator(){

    };
    
    
     double Entropy(double* p, unsigned int size);
    
    double getPolicyEntropy(unsigned int burn);
    
    

};


double FeatureInfoGainCalculator::getPolicyEntropy(unsigned int burn)
{
  double policyEntropy = 0.0;



  unsigned int numStates = curr_mdp->getNumStates();
  unsigned int numActions = curr_mdp->getNumActions();

  //for each state calculate entropy over chain
  double entropy_sum = 0.0;
  for(unsigned int s = 0; s < numStates; s++)
  {
    double opt_count = 0;
    double action_counts[numActions];
    for(int act = 0; act < numActions; act++)
      action_counts[act] = 0.0;

    //for each reward in chain count the number of times an action is optimal for that reward
    FeatureGridMDP** R_chain_base = birl->getRewardChain();
    for(unsigned int i= burn; i < chain_length; i++)
    {
      FeatureGridMDP* temp_mdp = R_chain_base[i];
      for(unsigned int a = 0; a < numActions; a++)
      {
        if(temp_mdp->isOptimalAction(s,a))
        {
          opt_count+=1.0;
          action_counts[a] += 1.0;
        }
      }
    }
    //calculate entropy over optimal actions at state s
    for(int a = 0; a < numActions; a++)
      action_counts[a] /= opt_count;
    double state_entropy = Entropy(action_counts, numActions);
    entropy_sum += state_entropy;
  }
  policyEntropy = entropy_sum / numStates;
  return policyEntropy;

}

double FeatureInfoGainCalculator::Entropy(double* p, unsigned int size)
{
  double entropy = 0.0;
  for(unsigned int i=0; i < size; i++)
  {
    if(p[i] != 0) entropy -= (p[i]*log(p[i]));
  }
  return entropy;
}


vector<double> getOptimalActionCount(unsigned int state, unsigned int numActions, FeatureMDP** rewardChain, unsigned int chain_length, int burn)
{

  vector<double> frequencies;
  for(unsigned int i = 0; i < numActions; i++) 
    frequencies.push_back(0);

  for(unsigned int i= burn; i < chain_length; i++)
  {
    FeatureMDP* temp_mdp = rewardChain[i];
    for(unsigned int a = 0; a < numActions; a++)
    {
        if(temp_mdp->isOptimalAction(state,a))
            frequencies[a] += 1;
    }
  }
  for(unsigned int i = 0; i < numActions; i++)
    frequencies[i] /= (chain_length - burn);
  return frequencies;

}



#endif
