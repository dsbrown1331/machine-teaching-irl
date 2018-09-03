
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
#include "feature_bandit_birl.hpp"


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
    FeatureBanditBIRL * base_birl = nullptr;
    FeatureBanditBIRL * good_birl = nullptr;
    FeatureBanditBIRL * bad_birl = nullptr;
    FeatureMDP* curr_mdp;
    double min_r, max_r, step_size, alpha;
    unsigned int chain_length;


  public:
    FeatureInfoGainCalculator(FeatureBanditBIRL* input_birl) {
      min_r = input_birl->getMinReward();
      max_r = input_birl->getMaxReward();
      curr_mdp = input_birl->getMDP();
      step_size = input_birl->getStepSize();
      chain_length = input_birl->getChainLength();
      alpha = input_birl->getAlpha();
      base_birl = input_birl; 
    };
    
    ~ FeatureInfoGainCalculator(){

    };
    
    
     long double Entropy(double* p, unsigned int size);
    
    long double getEntropy(pair<unsigned int,unsigned int> state_action, unsigned int K = 10);
    
    

};


long double FeatureInfoGainCalculator::getEntropy(pair<unsigned int,unsigned int> state_action, unsigned int K)
{
  long double info_gain = 0.0;

  unsigned int state = state_action.first;
  unsigned int action = state_action.second;

  long double probability_good = 0; 
  
  vector<double> frequencies;
  double total_frequency = 0;
  for(unsigned int i = 0; i < K; i++) frequencies.push_back(0);

  FeatureMDP** R_chain_base = base_birl->getRewardChain();
  for(unsigned int i= 50; i < chain_length; i++)
  {
    FeatureMDP* temp_mdp = R_chain_base[i];
    unsigned int numActions = temp_mdp->getNumActions();
    double Z0 [numActions]; 
    for(unsigned int a = 0; a < numActions; a++) Z0[a] = alpha*temp_mdp->getQValue(state,a);
    probability_good = exp(alpha*temp_mdp->getQValue(state,action) - base_birl->logsumexp(Z0,numActions));
    //cout <<"Ent prob good:" << probability_good << endl;
    for (unsigned int i = 0; i < K; i++)
    {
        if(probability_good > (double)i/K && probability_good <= (double)(i+1)/K )
        {
            frequencies[i] += 1;
            total_frequency += 1;
            break;
        }
    }
  }
  
  for(unsigned int i=0; i < K; i++)
  {
    if(frequencies[i] != 0){
     frequencies[i] /= total_frequency;
     info_gain += -(frequencies[i]*log(frequencies[i]));
     }
  }
  
  return info_gain;

}

long double FeatureInfoGainCalculator::Entropy(double* p, unsigned int size)
{
  long double entropy = 0.0;
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
