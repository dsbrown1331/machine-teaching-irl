#ifndef bandit_h
#define bandit_h

#include "mdp.hpp"


//make a multi-armed bandit problem for placement
class BanditMDP: public FeatureMDP {
    protected:
    
    
    public:
        BanditMDP(unsigned int num_arms, unsigned int nFeatures, double* fWeights, double** sFeatures): FeatureMDP(num_arms + 1, num_arms, nFeatures, fWeights, sFeatures, 1.0)
        {
            //only one initial state
            initialStates[0] = true;
            //every other state is terminal
            for(unsigned int i=1; i<num_arms + 1; i++)
            {
                terminalStates[i] = true; 
            }
            setBanditTransitions();
        }
        
        void displayFeatureWeights()
        {
            ios::fmtflags f( cout.flags() );
            std::streamsize ss = std::cout.precision();
         
            for(unsigned int f = 0; f < numFeatures; f++)
            {
                cout << setiosflags(ios::fixed)
                << setprecision(3)
                << featureWeights[f] << "  ";
            }
            cout << endl;

            cout.flags(f);
            cout << setprecision(ss);
        }
        
        void setBanditTransitions()
        {
            assert(T != nullptr);
            //make all transitions deterministic
            for(unsigned int a = 0; a < numActions; a++)
            {
            T[0][a][a+1] = 1.0;
            }
        }
        void displayValues()
        {
             ios::fmtflags f( cout.flags() );
            std::streamsize ss = std::cout.precision();

            assert(R != nullptr);
            //    if (R == nullptr){
            //      cout << "ERROR: no values!" << endl;
            //      return;
            //     }
            for(unsigned int s = 0; s < numStates; s++)
            {
                cout << setiosflags(ios::fixed)
                << setprecision(3)
                << V[s] << "  ";
            }
            cout << endl;

            cout.flags(f);
            cout << setprecision(ss);
        }
        void displayTransitions()
        {
            ios::fmtflags f( cout.flags() );
            std::streamsize ss = std::cout.precision();

            for(unsigned int a = 0; a < numActions; a++)
            {
                cout << "action " << a << endl;
                for(unsigned int s1 = 0; s1 < numStates; s1++)
                {
                    for(unsigned int s2 = 0; s2 < numStates; s2++)
                    {
                        cout << T[s1][a][s2] << " ";
                    }
                    cout << endl;
                }
            }
            cout.flags( f );
            cout << setprecision(ss);
        
        }
        void displayRewards()
        {
            ios::fmtflags f( cout.flags() );
            streamsize ss = std::cout.precision();
            assert(R != nullptr);
            cout << setiosflags(ios::fixed) << setprecision(2);
              
            for(unsigned int s = 0; s < numStates; s++)
            {
                cout << getReward(s) << "  ";
            }
            cout << endl;
            cout.flags(f);
            cout << setprecision(ss);
        }
        
        double getReward(unsigned int s) const
        { 
            assert(R != nullptr); //reward for state not defined

            return R[s];
        };
        
        void displayPolicy(vector<unsigned int> & policy)
        {
            cout << policy[0] << endl;
        }
    
};

#endif
