
#ifndef abbeel_projection_h
#define abbeel_projection_h

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
#include "mdp.hpp"
#include "confidence_bounds.hpp"


//TODO maybe write a wrapper for qvals that returns appropriately if not in map yet?

using namespace std;


double twoNormDiff(double x[], double y[], int length);

double twoNormDiff(double x[], double y[], int length)
      {
          double sum_squares = 0;
          for(int i = 0; i < length; i++)
          {
            double diff = x[i] - y[i];
            sum_squares += diff * diff;
          }
          return sqrt(sum_squares);
      };

class ProjectionIRL { // General gridMDP q-learner
      
   protected:
      FeatureGridMDP* fmdp;
      
      
      double* projectionStep(double mu[], double mu_bar[], double expertFcounts[]);
   public:
      ProjectionIRL(FeatureGridMDP* f): fmdp(f) {};
      ~ProjectionIRL(){
      
      };
      
      
      
      //main method to call to perform max-margin IRL
      vector<unsigned int> getProjectionPolicy(vector<unsigned int> & policy, vector<vector<pair<unsigned int,unsigned int> > > trajectories, double epsilon);
      
      
};

//algorithm step 2 from abbeel and ng
double* ProjectionIRL::projectionStep(double mu[], double mu_bar[], double mu_expert[])
{
    int numFeatures = fmdp->getNumFeatures();
    double* mu_bar_new = new double[numFeatures];
    
    //calculate mu - mu_bar
    double mu_diff_mu_bar[numFeatures];
    for(int i=0; i<numFeatures; i++)
        mu_diff_mu_bar[i] = mu[i] - mu_bar[i];
        
    //calculate mu_expert - mu_bar
    double expert_diff_mu_bar[numFeatures];
    for(int i=0;i < numFeatures; i++)
        expert_diff_mu_bar[i] =  mu_expert[i] - mu_bar[i];
        
        
    //calculate step_size
    double step_size = dotProduct(mu_diff_mu_bar, expert_diff_mu_bar, numFeatures) / dotProduct(mu_diff_mu_bar, mu_diff_mu_bar, numFeatures);
    
    //calculate new mu_bar
    for(int i=0; i<numFeatures; i++)
        mu_bar_new[i] = mu_bar[i] + step_size * mu_diff_mu_bar[i];
    
    return mu_bar_new;

}

vector<unsigned int> ProjectionIRL::getProjectionPolicy(vector<unsigned int> & policy, vector<vector<pair<unsigned int,unsigned int> > > trajectories, double epsilon)
{
    double eps = 0.001;
    //cout << "in getProjectionPolicy()" << endl;
    int numFeatures = fmdp->getNumFeatures();
    double mu[numFeatures];
    double mu_bar[numFeatures];
    double expertFcounts[numFeatures];
    
    //calculate expert feature counts
    double* demoFcounts = calculateEmpiricalExpectedFeatureCounts(trajectories, fmdp);
    for(int i=0; i<numFeatures; i++)
        expertFcounts[i] = demoFcounts[i];
    delete[] demoFcounts;
//    cout << "expert f counts" << endl;
//    for(unsigned int f = 0; f < numFeatures; f++)
//        cout << expertFcounts[f] << "\t";
//    cout << endl;
//    
    //Randomly pick a policy 
    fmdp->initRandDeterministicPolicy(policy);
    
    //compute mu_bar
    double* temp = calculateExpectedFeatureCounts(policy, fmdp, eps);
    for(int i=0;i<numFeatures;i++) 
        mu_bar[i] = temp[i];
    delete[] temp;
    
    //calculate new feature weights
    double fWeights[numFeatures];
    for(int i=0; i < numFeatures; i++)
        fWeights[i] = expertFcounts[i] - mu_bar[i];
        
    //solve for t
    double t = twoNormDiff(expertFcounts, mu_bar, numFeatures);
    
    //compute optimal policy using fWeights as new weights for mdp
    FeatureGridMDP temp_mdp(fmdp->getGridWidth(),fmdp->getGridHeight(), fmdp->getInitialStates(), fmdp->getTerminalStates(), fmdp->getNumFeatures(), fmdp->getFeatureWeights(), fmdp->getStateFeatures(), fmdp->isStochastic(), fmdp->getDiscount());
    temp_mdp.setFeatureWeights(fWeights);
    temp_mdp.valueIteration(eps);
    temp_mdp.getOptimalPolicy(policy);
    
    //compute expected feature counts of policy
    double* temp_fcnt = calculateExpectedFeatureCounts(policy, fmdp, eps);
    for(int i=0;i<numFeatures;i++) 
        mu[i] = temp_fcnt[i];
    delete[] temp_fcnt;
    double t_old = 0;
    int cnt = 0;    
    while(true)
    {
        t_old = t;
        cnt++;
        //cout << cnt << endl;
        //update mu_bar
        double* mu_bar_new = projectionStep(mu, mu_bar, expertFcounts);
        for(int i=0; i<numFeatures; i++)
            mu_bar[i] = mu_bar_new[i];
        delete[] mu_bar_new;
        
        //update fWeights
        for(int i=0; i < numFeatures; i++)
            fWeights[i] = expertFcounts[i] - mu_bar[i];
        
        //update t
        t = twoNormDiff(expertFcounts, mu_bar, numFeatures);
        //cout << "t = " << t << endl;
        if(t < epsilon)
        {
            cout << "converged" << endl;
            break;
        }
        if( cnt > 5000)
        {
            cout << "timed out" << endl;
            break;
        }
        //compute optimal policy using fWeights as new weights for mdp
        FeatureGridMDP temp_mdp(fmdp->getGridWidth(),fmdp->getGridHeight(), fmdp->getInitialStates(), fmdp->getTerminalStates(), fmdp->getNumFeatures(), fmdp->getFeatureWeights(), fmdp->getStateFeatures(), fmdp->isStochastic(), fmdp->getDiscount());
        temp_mdp.setFeatureWeights(fWeights);
        temp_mdp.valueIteration(eps);
        temp_mdp.getOptimalPolicy(policy);
        //temp_mdp.displayPolicy(policy);
        
        //compute expected feature counts of policy
        double* temp = calculateExpectedFeatureCounts(policy, fmdp, eps);
        for(int i=0;i<numFeatures;i++) 
            mu[i] = temp[i];
        delete[] temp;
        
    }
    
    return policy;

}

//Helper methods in confidence_bounds
//double** calculateStateExpectedFeatureCounts(vector<unsigned int> & policy, FeatureGridMDP* fmdp, double eps)

//Helper methods in mdp
//double dotProduct(double x[], double y[], int length)


#endif
