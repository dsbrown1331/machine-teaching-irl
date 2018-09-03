#ifndef machine_teaching_hpp
#define machine_teaching_hpp

double logsumexp(double* nums, unsigned int size);
double* computeQvalLogSumExps(double alpha, double** Qvals, unsigned int numStates, unsigned int numActions=4);
double computeLoss(vector<pair<unsigned int,unsigned int> > trajectory, double* Qlogsumexp, double alpha, double** Qvals, unsigned int numStates, unsigned int numActions = 4);
double computeLoss(pair<unsigned int,unsigned int> p, double* Qlogsumexp, double alpha, double** Qvals, unsigned int numStates, unsigned int numActions=4);

double logsumexp(double alpha, double* nums, unsigned int size) {
  double max_exp = alpha * nums[0];
  double sum = 0.0;
  unsigned int i;

  for (i = 1 ; i < size ; i++)
  {
    if (alpha * nums[i] > max_exp)
      max_exp = alpha * nums[i];
  }

  for (i = 0; i < size ; i++)
    sum += exp(alpha * nums[i] - max_exp);

  return log(sum) + max_exp;
}


double* computeQvalLogSumExps(double alpha, double** Qvals, unsigned int numStates, unsigned int numActions)
{
    double* stateQvalLSEs = new double[numStates];
    //for each states call logsumexp on Qvals for all actions in that state
    for(unsigned int i = 0; i < numStates; i++)
    {
        //cout << "Computing for state " << i << endl;
        //for(unsigned int j=0;j<numActions;j++)
        //    cout << Qvals[i][j] << ", ";
        //cout << endl;
        double val = logsumexp(alpha, Qvals[i], numActions);
        //cout << "log sum exp" << endl;
        //cout << val << endl; 
        stateQvalLSEs[i] = val;


    }
    return stateQvalLSEs;
}

double computeLoss(vector<pair<unsigned int,unsigned int> > trajectory, double* Qlogsumexp, double alpha, double** Qvals, unsigned int numStates, unsigned int numActions)
{
    double loss = 0.0;
    for(pair<unsigned int, unsigned int> p : trajectory)
    {
        unsigned int s = p.first;
        unsigned int a = p.second;
        //add in negative alpha * Qval
        loss -= alpha * Qvals[s][a];
        //add logsumexp over all actions in state s
        loss += Qlogsumexp[s];

    }
    return loss;
}

double computeLoss(pair<unsigned int,unsigned int> p, double* Qlogsumexp, double alpha, double** Qvals, unsigned int numStates, unsigned int numActions)
{
   double loss = 0.0;
   
    unsigned int s = p.first;
    unsigned int a = p.second;
    //add in negative alpha * Qval
    loss -= alpha * Qvals[s][a];
    //add logsumexp over all actions in state s
    loss += Qlogsumexp[s];


    return loss;

}

#endif
