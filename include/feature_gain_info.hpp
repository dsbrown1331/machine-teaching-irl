
#ifndef gain_h
#define gain_h

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <map>

#include <iostream>
#include <fstream>
#include <limits>

#include "../include/mdp.hpp"
#include "../include/feature_birl.hpp"


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
    FeatureBIRL * base_birl = nullptr;
    FeatureBIRL * good_birl = nullptr;
    FeatureBIRL * bad_birl = nullptr;
    FeatureGridMDP* curr_mdp;
    double min_r, max_r, step_size, alpha;
    unsigned int chain_length;


  public:
    FeatureInfoGainCalculator(FeatureBIRL* input_birl) {
      min_r = input_birl->getMinReward();
      max_r = input_birl->getMaxReward();
      curr_mdp = input_birl->getMDP();
      step_size = input_birl->getStepSize();
      chain_length = input_birl->getChainLength();
      alpha = input_birl->getAlpha();
      base_birl = input_birl; 
      good_birl = new FeatureBIRL(curr_mdp, min_r, max_r, chain_length, step_size, alpha);
      bad_birl  = new FeatureBIRL(curr_mdp, min_r, max_r, chain_length, step_size, alpha);
    };
    
    ~ FeatureInfoGainCalculator(){
      if(good_birl != nullptr) delete good_birl;
      if(bad_birl != nullptr) delete bad_birl;
    };
    
    long double getInfoGain(pair<unsigned int ,unsigned int> state_action);
    long double getInfoGainFromSamples(pair<unsigned int,unsigned int> state_action, double * prob_good);
    
    long double KLdivergence(long double* p, long double* q, unsigned int size);
    long double KNN_KLdivergence(FeatureGridMDP** p, FeatureGridMDP** q, unsigned int size);
    
     long double Entropy(double* p, unsigned int size);
    long double JSdivergence(long double* p, long double* q, unsigned int size);
    void sortAndWriteToFile(long double * base_posterior,long double * good_posterior, long double* bad_posterior);
    
    long double getEntropy(pair<unsigned int,unsigned int> state_action, int K = 10);
    

    

};

long double FeatureInfoGainCalculator::getEntropy(pair<unsigned int,unsigned int> state_action, int K)
{
  long double info_gain = 0.0;

  unsigned int state = state_action.first;
  unsigned int action = state_action.second;

  long double probability_good = 0; 
  
  vector<double> frequencies;
  double total_frequency = 0;
  for(unsigned int i = 0; i < K; i++) frequencies.push_back(0);

  FeatureGridMDP** R_chain_base = base_birl->getRewardChain();
  for(unsigned int i= 50; i < chain_length; i++)
  {
    FeatureGridMDP* temp_mdp = R_chain_base[i];
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


long double FeatureInfoGainCalculator::getInfoGainFromSamples(pair<unsigned int,unsigned int> state_action, double * prob_good)
{
  long double info_gain = 0.0;

  unsigned int state = state_action.first;
  unsigned int action = state_action.second;

  good_birl->addPositiveDemos(base_birl->getPositiveDemos());
  good_birl->addNegativeDemos(base_birl->getNegativeDemos());
  good_birl->addPositiveDemo(state_action);

  good_birl->run();

  bad_birl->addPositiveDemos(base_birl->getPositiveDemos());
  bad_birl->addNegativeDemos(base_birl->getNegativeDemos());
  bad_birl->addNegativeDemo(state_action);

  bad_birl->run();

  FeatureGridMDP** R_chain_base = base_birl->getRewardChain();
  FeatureGridMDP** R_chain_good = good_birl->getRewardChain();
  FeatureGridMDP** R_chain_bad  = bad_birl->getRewardChain();
  
  long double probability_good = 0;
  for(unsigned int i=0; i < chain_length; i++)
  {
    MDP* temp_mdp = R_chain_base[i];
    unsigned int numActions = temp_mdp->getNumActions();

    double Z0 [numActions]; 
    for(unsigned int a = 0; a < numActions; a++) Z0[a] = alpha*temp_mdp->getQValue(state,a);
    probability_good += exp(alpha*temp_mdp->getQValue(state,action) - base_birl->logsumexp(Z0,numActions));
  }
  probability_good /= chain_length;
  //cout << "  - probability_good: " << probability_good <<  endl;
  *prob_good = probability_good;
  long double divergence_good = KNN_KLdivergence(R_chain_base, R_chain_good, chain_length);
  long double divergence_bad  = KNN_KLdivergence(R_chain_base, R_chain_bad, chain_length);
  
  //cout << "  - divergence: " << divergence_good << ", " << divergence_bad <<  endl;
  
  info_gain = divergence_good*probability_good + (1 - probability_good)*divergence_bad;
  
  good_birl->removeAllDemostrations();
  bad_birl->removeAllDemostrations();
  return info_gain;

}


long double FeatureInfoGainCalculator::getInfoGain(pair<unsigned int,unsigned int> state_action)
{
  long double info_gain = 0.0;

  unsigned int state = state_action.first;
  unsigned int action = state_action.second;

  good_birl->addPositiveDemos(base_birl->getPositiveDemos());
  good_birl->addNegativeDemos(base_birl->getNegativeDemos());
  good_birl->addPositiveDemo(state_action);


  good_birl->run();

  bad_birl->addPositiveDemos(base_birl->getPositiveDemos());
  bad_birl->addNegativeDemos(base_birl->getNegativeDemos());
  bad_birl->addNegativeDemo(state_action);


  bad_birl->run();

  FeatureGridMDP** R_chain_base = base_birl->getRewardChain();
  FeatureGridMDP** R_chain_good = good_birl->getRewardChain();
  FeatureGridMDP** R_chain_bad  = bad_birl->getRewardChain();
  
  long double probability_good = 0;
  for(unsigned int i=0; i < chain_length; i++)
  {
    MDP* temp_mdp = R_chain_base[i];
    unsigned int numActions = temp_mdp->getNumActions();

    double Z0 [numActions]; 
    for(unsigned int a = 0; a < numActions; a++) Z0[a] = alpha*temp_mdp->getQValue(state,a);
    probability_good += exp(alpha*temp_mdp->getQValue(state,action) - base_birl->logsumexp(Z0,numActions));
  }
  probability_good /= chain_length;
  //cout << "  - probability_good: " << probability_good <<  endl;
  long double divergence_good = KNN_KLdivergence(R_chain_base, R_chain_good, chain_length);
  long double divergence_bad  = KNN_KLdivergence(R_chain_base, R_chain_bad, chain_length);
  
  //cout << "  - divergence: " << divergence_good << ", " << divergence_bad <<  endl;
  
  info_gain = divergence_good*probability_good + (1 - probability_good)*divergence_bad;
  
  good_birl->removeAllDemostrations();
  bad_birl->removeAllDemostrations();
  return info_gain;
  /*long double info_gain = 0.0;

  unsigned int state = state_action.first;
  unsigned int action = state_action.second;
  
  unsigned int total_actions = curr_mdp->getNumStates()*curr_mdp->getNumActions();
  
  double current_alpha = alpha ; // (1 + base_birl->getNumDemonstrations()/total_actions);
  
   //probability of (s,a) being good or bad given current distribution
  
  good_birl->addPositiveDemos(base_birl->getPositiveDemos());
  good_birl->addNegativeDemos(base_birl->getNegativeDemos());
  good_birl->addPositiveDemo(state_action);
  good_birl->setAlpha(current_alpha);

  good_birl->run();

  
  bad_birl->addPositiveDemos(base_birl->getPositiveDemos());
  bad_birl->addNegativeDemos(base_birl->getNegativeDemos());
  bad_birl->addNegativeDemo(state_action);
  bad_birl->setAlpha(current_alpha);

  bad_birl->run();

  FeatureGridMDP* R_chain2 [chain_length];

  FeatureGridMDP** R_chain_base = base_birl->getRewardChain();
  FeatureGridMDP** R_chain_good = good_birl->getRewardChain();
  FeatureGridMDP** R_chain_bad  = bad_birl->getRewardChain();

  for(unsigned int idx = 0; idx < chain_length; idx+=3)  R_chain2[idx/3] =  R_chain_base[idx];    
  for(unsigned int idx = 0; idx < chain_length; idx+=3)  R_chain2[(idx + chain_length)/3] =  R_chain_good[idx]; 
  for(unsigned int idx = 0; idx < chain_length; idx+=3)  R_chain2[(idx + chain_length*2)/3] =  R_chain_bad[idx]; 
 

  // Union (not worth the effort to remove duplicates!)

  long double base_posterior[chain_length];
  long double good_posterior[chain_length];
  long double bad_posterior[chain_length];

  long double sum = 0; //sum of base posteriors

  double* curr_posterior = base_birl->getPosteriorChain();
  
  
  for(unsigned int idx=0; idx < chain_length; idx+=3){
    base_posterior[idx/3] = curr_posterior[idx];
    sum += base_posterior[idx/3];
  }
  
  for(unsigned int idx=0; idx < chain_length; idx+=3){
    base_posterior[(idx+chain_length)/3] = exp(base_birl->calculatePosterior(R_chain_good[idx]));
    sum += base_posterior[(idx+chain_length)/3];
  }
  
  for(unsigned int idx=0; idx < chain_length; idx+=3){
    base_posterior[(idx+2*chain_length)/3] = exp(base_birl->calculatePosterior(R_chain_bad[idx]));
    sum += base_posterior[(idx+2*chain_length)/3];
  }
    
  
  long double sum_good = 0;
  long double sum_bad = 0;

  curr_posterior = good_birl->getPosteriorChain();
  for(unsigned int idx=0; idx < chain_length; idx+=3){
    good_posterior[(idx+chain_length)/3] = curr_posterior[idx];
    sum_good += good_posterior[(idx+chain_length)/3];
  }
  for(unsigned int idx=0; idx < chain_length/3; idx++){ 
    good_posterior[idx] = exp(good_birl->calculatePosterior(R_chain2[idx]));
    sum_good += good_posterior[idx];
  }
  for(unsigned int idx=0; idx < chain_length; idx+=3){ 
    good_posterior[(idx+2*chain_length)/3] = exp(good_birl->calculatePosterior(R_chain_bad[idx]));
    sum_good += good_posterior[(idx+2*chain_length)/3];
  }

  curr_posterior = bad_birl->getPosteriorChain();
  for(unsigned int idx=0; idx < chain_length; idx+=3){
    bad_posterior[(idx+2*chain_length)/3] = curr_posterior[idx];
    sum_bad += bad_posterior[(idx+2*chain_length)/3];
  }
  for(unsigned int idx=0; idx < chain_length/3; idx++){
    bad_posterior[idx] = exp(bad_birl->calculatePosterior(R_chain2[idx]));
    sum_bad +=  bad_posterior[idx];
  }
  for(unsigned int idx=0; idx < chain_length; idx+=3){
    bad_posterior[(idx+chain_length)/3] = exp(bad_birl->calculatePosterior(R_chain_good[idx]));
    sum_bad += bad_posterior[(idx+chain_length)/3];
  }

  for(unsigned int idx=0; idx < chain_length; idx++){
    if( sum > 0) base_posterior[idx] /= sum;
    if( sum_good > 0) good_posterior[idx] /= sum_good;
    if( sum_bad > 0) bad_posterior[idx]  /= sum_bad;

  }
  
   
  long double probability_good = 0; //, probability_bad = 0;
  for(unsigned int i=0; i < chain_length; i++)
  {
    MDP* temp_mdp = R_chain2[i];
    unsigned int numActions = temp_mdp->getNumActions();

    double Z0 [numActions]; 
    for(unsigned int a = 0; a < numActions; a++) Z0[a] = current_alpha*temp_mdp->getQValue(state,a);
    probability_good += base_posterior[i]*exp(current_alpha*temp_mdp->getQValue(state,action) - base_birl->logsumexp(Z0,numActions));
  }
  //cout << "  - probability (g,b): " << probability_good << ", " << (1 - probability_good) <<  endl;
  
  if( probability_good > 0.1 && probability_good < 0.9){

  long double divergence_good = KLdivergence(base_posterior, good_posterior, chain_length);
  long double divergence_bad  = KLdivergence(base_posterior, bad_posterior, chain_length);
  
  
  //cout << "  - divergence: " << divergence_good << ", " << divergence_bad <<  endl;
  
    info_gain = probability_good*(divergence_good+base_birl->getNumDemonstrations()/total_actions) + (1 - probability_good)*divergence_bad;
  }
  else info_gain = 0;
  
   //cout << "  Info Gain: " << info_gain << endl;
  return info_gain;*/

}

long double FeatureInfoGainCalculator::KNN_KLdivergence(FeatureGridMDP** p, FeatureGridMDP** q, unsigned int size)
{
  long double divergence = 0.0;
  //long double max_value = numeric_limits<double>::max() / (size+1);
  long double c = (long double)(p[0]->getNumStates()) / size;
    
  for(unsigned int i=0; i < size; i++)
  {
    //if(i % 100 == 0) cout << p[i] << ";" << q[i] << " ; " << p[i]*log(p[i]/q[i]) << endl;
    //if(q[i] > 0.00000001 && p[i] != 0) divergence += (p[i]*log(p[i]/q[i]));
    //else if(p[i] != 0) divergence += (p[i]*log(p[i]*100000000));
    long double dist_pp = numeric_limits<double>::infinity();
    long double dist_pq = numeric_limits<double>::infinity();
    
    for(unsigned int j=0; j < size; j++)
    {
        if(j != i)  
        {
            long double dist_ij = p[i]->L2_distance(p[j]);
            if (dist_ij < dist_pp )  dist_pp = dist_ij;
        }
    }
    
    for(unsigned int j=0; j < size; j++)
    {
        long double dist_ij = p[i]->L2_distance(q[j]);
        if (dist_ij < dist_pq)  dist_pq = dist_ij;
    }
    
    dist_pp = max(dist_pp, 0.000001l);
    dist_pq = min(dist_pq, 10000.0l);
    divergence += log((dist_pq/dist_pp)/1000+1);
    //cout << i << " dists:" << dist_pp << "," << dist_pq << "," << log(dist_pq/dist_pp) << " divergence:" << divergence << endl;
    if (divergence < 0){ // overflow?
       divergence = 100000.0l;
       break;
    }  
  }
  
  
  divergence = c*(divergence+log(1000)) + log(size/(size-1));
  return divergence;
}

long double FeatureInfoGainCalculator::KLdivergence(long double* p, long double* q, unsigned int size)
{
  long double divergence = 0.0;
  for(unsigned int i=0; i < size; i++)
  {
    //if(i % 100 == 0) cout << p[i] << ";" << q[i] << " ; " << p[i]*log(p[i]/q[i]) << endl;
    if(q[i] > 0.00000001 && p[i] != 0) divergence += (p[i]*log(p[i]/q[i]));
    else if(p[i] != 0) divergence += (p[i]*log(p[i]*100000000));
  }

  return divergence;
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

long double FeatureInfoGainCalculator::JSdivergence(long double* p, long double* q, unsigned int size)
{
  long double divergence = 0.0;   

  for(unsigned int i=0; i < size; i++)
  {
    long double avg = (p[i] + q[i]) /2;
    //cout << p[i] << "," << q[i] << "," << avg << "," << 0.5*(p[i]*log(p[i]/avg)) + 0.5*(q[i]*log(q[i]/avg)) << endl;
    if     (avg != 0 && p[i] == 0 && q[i] != 0) divergence += 0.5*(q[i]*log(q[i]/avg));
    else if(avg != 0 && p[i] != 0 && q[i] == 0) divergence += 0.5*(p[i]*log(p[i]/avg));
    else if(avg != 0 && p[i] != 0 && q[i] != 0) divergence += 0.5*(p[i]*log(p[i]/avg)) + 0.5*(q[i]*log(q[i]/avg));
  }

  return divergence;
}

void FeatureInfoGainCalculator::sortAndWriteToFile(long double * base_posterior,long double * good_posterior, long double* bad_posterior)
{
  //for plotting purpose
  vector<pair<long double, long double> > zipped_bg, zipped_bb;
  zip(base_posterior, good_posterior, chain_length, zipped_bg);
  zip(base_posterior, bad_posterior,  chain_length, zipped_bb);

  sort(zipped_bg.begin(), zipped_bg.end(), greaterThan);
  sort(zipped_bb.begin(), zipped_bb.end(), greaterThan);

  unzip(zipped_bg, base_posterior, good_posterior);
  unzip(zipped_bb, base_posterior, bad_posterior);

  //printing for plot
  //cout << "\n--- printing for plot ---" << endl;
  //sort(base_posterior.begin(), base_posterior.end());
  ofstream basefile;
  basefile.open ("data/base.dat");
  for(unsigned int idx=0; idx < chain_length; idx++){
    basefile << idx << " " << base_posterior[idx] << endl;
  }
  basefile.close();

  //sort(good_posterior.begin(), good_posterior.end());
  ofstream goodfile;
  goodfile.open ("data/good.dat");
  for(unsigned int idx=0; idx < chain_length; idx++){
    goodfile << idx << " " << good_posterior[idx] << endl;
  }
  goodfile.close();

  //sort(bad_posterior.begin(), bad_posterior.end());
  ofstream badfile;
  badfile.open ("data/bad.dat");
  for(unsigned int idx=0; idx < chain_length; idx++){
    badfile << idx << " " << bad_posterior[idx] << endl;
  }
  badfile.close();
}


#endif
