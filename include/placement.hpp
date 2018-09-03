#ifndef placement_h
#define placement_h

#include "mdp.hpp"

class PlacementMDP: public MDP {
    protected:
    
    
    public:
        PlacementMDP(unsigned int num_placements, double gamma): MDP(gamma, states, 2)
        {
            cout << "num states : " << numStates << endl;
            cout << "num actions: " << numActions << endl;
            for(unsigned int i=0; i<initStates.size(); i++)
            {
                int idx = initStates[i];
                initialStates[idx] = true;
            }
            for(unsigned int i=0; i<termStates.size(); i++)
            {
                int idx = termStates[i];
                terminalStates[idx] = true; 
            }
            setDeterministicChainTransitions();
        }
        
        ChainMDP(unsigned int states, bool* initStates, bool* termStates, double gamma=0.95): MDP(gamma, states, 2)
        { 
            for(unsigned int i=0; i<numStates; i++)
            {
                initialStates[i] = initStates[i];
            }
            for(unsigned int i=0; i<numStates; i++)
            {
                terminalStates[i] = termStates[i]; 
            }
            setDeterministicChainTransitions();
        }
        void setDeterministicChainTransitions();
        void displayValues();
        void displayTransitions();
        void displayRewards();
        
        double getReward(unsigned int s) const
        { 
            assert(R != nullptr); //reward for state not defined

            return R[s];
        };
        
        void displayPolicy(vector<unsigned int> & policy)
        {

            for(unsigned int s = 0; s < numStates; s++)
            {
                if(isTerminalState(s)) 
                    cout << "*" << "  ";
                else if(policy[s]==0) 
                    cout << "<" << "  ";
                else  
                    cout << ">" << "  ";
           
            }
            cout << endl;
        }
    
};

void ChainMDP::displayRewards()
{
    ios::fmtflags f( cout.flags() );
    streamsize ss = std::cout.precision();
    assert(R != nullptr);
//    if (R == nullptr){
//      cout << "ERROR: no rewards!" << endl;
//      return;
//     }
    cout << setiosflags(ios::fixed) << setprecision(2);
          
    for(unsigned int s = 0; s < numStates; s++)
    {
        cout << getReward(s) << "  ";
    }
    cout << endl;
    cout.flags(f);
    cout << setprecision(ss);
};

void ChainMDP::displayValues()
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
};


void ChainMDP::setDeterministicChainTransitions() //specific to markov chain
{
    assert(T != nullptr);
    //LEFT
    for(unsigned int s = 0; s < numStates; s++)
    {

        if( s > 0)
        {
            T[s][LEFT][s - 1] = 1.0;
        }
        else
        { 
            T[s][LEFT][s] = 1.0;
        }

    }
         
    //RIGHT
    for(unsigned int s = 0; s < numStates; s++)
    {
        if(s + 1 < numStates)
            T[s][RIGHT][s + 1] = 1.0;
        else
            T[s][RIGHT][s] = 1.0;

    }
    
    //Terminals
    if(terminalStates != nullptr) 
    {
        //cout << "setting up terminals" << endl;
        for(unsigned int s = 0; s < numStates; s++)
        {
           if(terminalStates[s])
           {
               if(s  > 0) 
                   T[s][LEFT][s - 1] = 0.0;
               if(s + 1 < numStates) 
                   T[s][RIGHT][s + 1] = 0.0;
               T[s][LEFT][s] = 0.0; 
               T[s][RIGHT][s] = 0.0;
           }
        }
    }
  
   

            
    //check that all state transitions add up properly!
    for(unsigned int s = 0; s < numStates; s++)
    {
        //cout << "state " << s << endl;
        for(unsigned int a = 0; a < numActions; a++)
        {
            //cout << "action " << a << endl;
            //add up transitions
            double sum = 0;
            for(unsigned int s2 = 0; s2 < numStates; s2++)
            
                    sum += T[s][a][s2];
            //cout << sum << endl;
            assert(sum <= 1.0);
        }
    }

};


void ChainMDP::displayTransitions()
{
   ios::fmtflags f( cout.flags() );
   std::streamsize ss = std::cout.precision();

    cout << "-------- LEFT ----------" << endl;
    for(unsigned int s1 = 0; s1 < numStates; s1++)
    {
        for(unsigned int s2 = 0; s2 < numStates; s2++)
        {
            cout << T[s1][LEFT][s2] << " ";
        }
        cout << endl;
    }
    cout << "-------- RIGHT ----------" << endl;
    for(unsigned int s1 = 0; s1 < numStates; s1++)
    {
        for(unsigned int s2 = 0; s2 < numStates; s2++)
        {
            cout << T[s1][RIGHT][s2] << " ";
        }
        cout << endl;
    }
    cout.flags( f );
    cout << setprecision(ss);
}
