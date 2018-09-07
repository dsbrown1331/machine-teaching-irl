#include <iostream>
#include <algorithm>
#include "../include/mdp.hpp"
#include "../include/grid_domains.hpp"
#include "../include/optimalTeaching.hpp"
#include "../include/confidence_bounds.hpp"
#include "../include/feature_birl.hpp"
#include <fstream>
#include <string>

using namespace std;

//TODO: get it to report Policy loss
//Get it to run multiple iterations and write to file for later averaging

vector<unsigned int> getMAPBirlPolicy(FeatureGridMDP *fmdp, vector<pair<unsigned int,unsigned int> > demos, double precision)
{

    double alpha = 100; //50                    //confidence param for BIRL
    unsigned int chain_length = 20000;//10000;//1000;//5000;        //length of MCMC chain
    int sample_flag = 4;                      //param for mcmc walk type
    int num_steps = 10;                       //tweaks per step in mcmc
    bool mcmc_reject_flag = true;             //allow for rejects and keep old weights
    double step = 0.001; //0.01
    const double min_r = -1;
    const double max_r = 1;

    FeatureBIRL birl(fmdp, min_r, max_r, chain_length, step, alpha, sample_flag, mcmc_reject_flag, num_steps);
    birl.addPositiveDemos(demos);
    birl.displayDemos();
    birl.run();
    FeatureGridMDP* mapMDP = birl.getMAPmdp();
    mapMDP->displayFeatureWeights();
    //cout << "Recovered reward" << endl;
    //mapMDP->displayRewards();

    //solve for the optimal policy
    vector<unsigned int> eval_pi (mapMDP->getNumStates());
    mapMDP->valueIteration(precision);
    cout <<"--wall states--" << endl;
    for(unsigned int s = 0; s < mapMDP->getNumStates(); s++)
        cout << mapMDP->isWallState(s) << ",";
    cout << endl;
    cout << "-- value function ==" << endl;
    mapMDP->displayValues();
    mapMDP->calculateQValues();
    mapMDP->getOptimalPolicy(eval_pi);
    cout << "Recovered reward" << endl;
    mapMDP->displayRewards();
    cout << "-- optimal policy --" << endl;
    mapMDP->displayPolicy(eval_pi);
    
    return eval_pi;
}

vector<pair<unsigned int, unsigned int> > processTrajectories(vector<vector<pair<unsigned int,unsigned int> > > trajectories, bool removeDuplicates)
{
    vector<pair<unsigned int,unsigned int> >  demos;
    for(vector<pair<unsigned int, unsigned int> > traj : trajectories)
    {
        for(pair<unsigned int, unsigned int> p : traj)
        {
            if(removeDuplicates)
            {
                if(std::find(demos.begin(), demos.end(), p) == demos.end())
                    demos.push_back(p);
            }
            else
            {    
                demos.push_back(p);
            }
        }
    }
    return demos;
}


int main(int argc, char *argv[]) 
{

    if(argc < 4)
    {
        cout << "Usage: ./cakmakTasksExperiment algo_name task trajLength [numSamples]\n where [numSamples] is only required for cakmak" << endl;
        exit(EXIT_FAILURE);
    }
    string algo = argv[1];
    string world = argv[2]; //TODO change this programatically
    string traj_length = argv[3];

    int numSamples = 0; //for Cakmak algo
    if(argc == 5)
        numSamples = stoi(argv[4]);

    
    cout << world << endl;
    
    int seed = 231234;

    
    //srand(123);
    //test arrays to get features
    int K = 20;  //how many rollouts from each initial state to try
    int N = 9;
    int F_l = 8;
    int F_u = 8;
    int F_step = 1;
    int trajLength = stoi(traj_length);
    
    double eps = 0.00; //2^3
    double precision = 1E-6;//0.000;
    bool removeDuplicates = true;
    int replicates = 20;//30;
    bool stochastic = true;
    
    
    for(int F = F_l; F <= F_u; F+=F_step)
    {
        
        
        string params = "";
        params += "N="+to_string(N)+"_F="+to_string(F);
        if(algo == "cakmak")
            params += "_samp" + to_string(numSamples);
        string out_file_path = "data/algorithm_comparison/" + algo + "_" + params + "_" + world + "_" + "trajLength=" + traj_length + ".txt";
        cout << out_file_path << endl;
        ofstream outfile(out_file_path);
        
            
        
        for(int rep = 14; rep < replicates; rep++)
        {
            srand(seed + 13 * rep);


        //    FeatureGridMDP* task = generateStochasticDebugMDP();
            FeatureMDP* task = NULL;
            if(world == "random")
                task = generateRandomWorldNStates(F, N);
            else if(world == "random_mixed")
                task = generateRandomWorldNStatesMixedFeatures(F, N);
            else
            {
                cout << "bad task name" << endl;
                exit(EXIT_FAILURE);
                
            }

                
            //FeatureGridMDP* task = generateCakmakBadExample3();
            
            
            task->valueIteration(precision);
            task->calculateQValues();
            vector<unsigned int> eval_pi (task->getNumStates());
            task->getOptimalPolicy(eval_pi);
            cout <<"--wall states--" << endl;
            //bool* wallStates = task->getWallStates();
            //for(unsigned int s = 0; s < task->getNumStates(); s++)
            //    cout << wallStates[s] << ",";
            cout << endl;
            cout << "-- optimal deterministic policy --" << endl;
            task->displayPolicy(eval_pi);
            vector< vector<double> > opt_policy = task->getOptimalStochasticPolicy();
            cout << "-- reward function --" << endl;
            task->displayRewards();
            //cout << "-- value function ==" << endl;
            //task->displayValues();
        //    cout << "-- optimal stochastic policy --" << endl;
        //    for(unsigned int s=0; s < task->getNumStates(); s++)
        //    {
        //        if(!task->isWallState(s))
        //        {
        //            cout << "state " << s << ": ";
        //            for(unsigned int a = 0; a < task->getNumActions(); a++)
        //                cout << opt_policy[s][a] << ",";
        //            cout << endl;
        //        }
        //    }
            
        

            cout << "========== REP " << rep << "==============" << endl;
            cout << "World = " << world << endl;

            clock_t c_start = clock();
           
            vector<vector<pair<unsigned int,unsigned int> > > trajectories;
            
                        
            if(algo == "cakmak")
            {
                //Cakmak algorithm
                cout << "using Cakmak algorithm" << endl;
                trajectories = OptimalTeachingIRL_Cakmak_Vectorized(task, K, trajLength, numSamples, eps, precision, stochastic);
            }
            else if(algo == "baseline")
            {
                //set-cover baseline
                cout << "using baseline SCOT algorithm without redundancy removal" << endl;
                trajectories = solveSetCoverBaseline(task, K, trajLength);
            }
            else if(algo == "scot")
            {
                //set-cover randomized without redundancies
                cout << "using SCOT algorithm" << endl;
                trajectories= solveSetCoverOptimalTeaching(task, K, trajLength,precision, stochastic); 
            }
            else
            {
                cout << "not a valid algorithm name" << endl;
                return 1;
            }
            
            clock_t c_end = clock();

           //learn policy from demonstrations
           vector<pair<unsigned int, unsigned int> > demos = processTrajectories(trajectories, removeDuplicates);
           vector<unsigned int> policy = getMAPBirlPolicy((FeatureGridMDP*)task, demos, precision);
           double loss = getPolicyLoss(task, policy, precision);
           double zero_one_loss = policyLoss(policy, task);
           double time_elapsed = (c_end-c_start)*1.0 / CLOCKS_PER_SEC;
           int num_trajectories = trajectories.size();
           int num_sa_pairs = demos.size();
           cout << "\n[Timing] Time passed: " << time_elapsed << " s\n"; 
           cout << "Num trajectories = " << num_trajectories << endl;
           cout << "Num unique state-action pairs = " << num_sa_pairs << endl;
	        cout << " policy loss = " << loss << endl;
	        cout << " 0/1 loss = " << zero_one_loss << endl;
	
	        outfile << time_elapsed << "," << num_trajectories << "," << num_sa_pairs
                    << "," << loss << "," << zero_one_loss << endl;
	
            cout << "---- Solution ------" << endl;
            int count = 0;
            //print out demos
            for(vector<pair<unsigned int, unsigned int> > traj : trajectories)
            {
                cout << "demo " << count << endl;
                count++;
                for(pair<unsigned int, unsigned int> p : traj)
                    cout << "(" <<  p.first << "," << p.second << "), ";
                cout << endl;
            }
            
            //clean up memory
            //first delete features
            double** stateFeatures = task->getStateFeatures();
            for(unsigned int s1 = 0; s1 < task->getNumStates(); s1++)
            {
                delete[] stateFeatures[s1];
            }
            delete[] stateFeatures;
            stateFeatures = NULL;
            //then delete mdp
            delete task;
            task = NULL;
        }
        outfile.close();

    }

}


