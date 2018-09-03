#ifndef optimalTeaching_h
#define optimalTeaching_h

#include "confidence_bounds.hpp"
#include "unit_norm_sampling.hpp"
#include "grid_domains.hpp"
#include <map>
#include <cmath>
#include <list>
#include <math.h>
#include <assert.h>
#include <cmath>
#include "lp_helper.hpp"
#include <algorithm>
#include <Eigen/Dense>

using namespace Eigen;

bool state_action_comp (pair<unsigned int, unsigned int> a, pair<unsigned int, unsigned int> b)
{
    if(a.first < b.first) 
        return true;
    else if (a.first == b.first)
    {
        return a.second < b.second;
    }
    else 
        return false;
    
}

bool areEqual(vector<double> a, vector<double> b);
vector<vector<pair<unsigned int, unsigned int > > > solveSetCoverOptimalTeaching(FeatureMDP* fmdp, int K, int horizon, double eps, bool stochastic=true);

void printOutDemos(vector<vector<pair<unsigned int, unsigned int> > > demos)
{
 //print out demos
    int count = 0;
    for(vector<pair<unsigned int, unsigned int> > traj : demos)
    {
        cout << "demo " << count << endl;
        count++;
        for(pair<unsigned int, unsigned int> p : traj)
            cout << "(" <<  p.first << "," << p.second << "), ";
        cout << endl;
    }
}

FeatureMDP* generateRandom9x9World()
{
    unsigned int numStates = 81;
    unsigned int numFeatures = 8;
    const int width = 9;
    const int height = 9;
    double featureWeights[numFeatures];
    double* randWeights = sample_unit_L1_norm(numFeatures);
    for(unsigned int i=0;i<numFeatures;i++)
        featureWeights[i] = randWeights[i];
    delete randWeights;
        
    double gamma = 0.95;
    double** stateFeatures = random9x9GridNavWorld8Features();
    
    //set up terminals and inits
    vector<unsigned int> initStates(numStates);
    for(unsigned int s = 0; s < numStates; s++)
        initStates[s] = s;
        

    vector<unsigned int> termStates;
    vector<unsigned int> wallStates;
    bool stochastic = false;
   
    FeatureMDP* world1 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world1)->setWallState(s);
        

    return world1;

}

    

FeatureMDP* generateRandom9x9World(unsigned int f)
{
    unsigned int numStates = 81;
    unsigned int numFeatures = f;
    unsigned int size = 9;
    const int width = size;
    const int height = size;
    double featureWeights[numFeatures];
    double* randWeights = sample_unit_L1_norm(numFeatures);
    for(unsigned int i=0;i<numFeatures;i++)
        featureWeights[i] = randWeights[i];
    delete randWeights;
        
    double gamma = 0.95;
    double** stateFeatures = randomNxNGridNavWorldXFeatures(size, f);
    
    //set up terminals and inits
    vector<unsigned int> initStates(numStates);
    for(unsigned int s = 0; s < numStates; s++)
        initStates[s] = s;
        

    vector<unsigned int> termStates;
    //vector<unsigned int> wallStates;
    bool stochastic = false;
   
    FeatureMDP* world1 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    //for(unsigned int s : wallStates)
    //    ((FeatureGridMDP*) world1)->setWallState(s);
        

    return world1;

}

FeatureMDP* generateRandomWorld8Features(int f, int N)
{
    assert(f == 8);
    unsigned int numStates = N*N;
    unsigned int numFeatures = f;
    const int width = N;
    const int height = N;
    double featureWeights[numFeatures];
    double* randWeights = sample_unit_L1_norm(numFeatures);
    for(unsigned int i=0;i<numFeatures;i++)
        featureWeights[i] = randWeights[i];
    delete randWeights;
        
    double gamma = 0.95;
    double** stateFeatures = randomNxNGridNavWorld8Features(N);
    
    //set up terminals and inits
    vector<unsigned int> initStates(numStates);
    for(unsigned int s = 0; s < numStates; s++)
        initStates[s] = s;
        

    vector<unsigned int> termStates;
    vector<unsigned int> wallStates;
    bool stochastic = false;
   
    FeatureMDP* world1 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world1)->setWallState(s);
        

    return world1;

}


FeatureMDP* generateRandomWorldNStates(int f, int N)
{
    //assert(N == 10);
    unsigned int numStates = N*N;
    unsigned int numFeatures = f;
    const int width = N;
    const int height = N;
    double featureWeights[numFeatures];
    double* randWeights = sample_unit_L1_norm(numFeatures);
    for(unsigned int i=0;i<numFeatures;i++)
        featureWeights[i] = randWeights[i];
    delete [] randWeights;
        
    double gamma = 0.95;
    double** stateFeatures = randomNxNGridNavWorldXFeatures(N, f);
    
    //set up terminals and inits
    vector<unsigned int> initStates(numStates);
    for(unsigned int s = 0; s < numStates; s++)
        initStates[s] = s;
        

    vector<unsigned int> termStates;
    vector<unsigned int> wallStates;
    bool stochastic = false;
   
    FeatureMDP* world1 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world1)->setWallState(s);
        

    return world1;

}


FeatureMDP* generateRandomWorldNStatesMixedFeatures(int f, int N)
{
    //assert(N == 10);
    unsigned int numStates = N*N;
    unsigned int numFeatures = f;
    const int width = N;
    const int height = N;
    double featureWeights[numFeatures];
    double* randWeights = sample_unit_L1_norm(numFeatures);
    for(unsigned int i=0;i<numFeatures;i++)
        featureWeights[i] = randWeights[i];
    delete [] randWeights;
        
    double gamma = 0.95;
    double** stateFeatures = randomNxNGridNavWorldMixedFeatures(N, f);
    
    //set up terminals and inits
    vector<unsigned int> initStates(numStates);
    for(unsigned int s = 0; s < numStates; s++)
        initStates[s] = s;
        

    vector<unsigned int> termStates;
    vector<unsigned int> wallStates;
    bool stochastic = false;
   
    FeatureMDP* world1 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world1)->setWallState(s);
        

    return world1;

}




//return fully formed cakmak world 1 with task 1 weights
FeatureMDP* generateCakmakTask1bMAPReward()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 7;
    const int height = 6;
    const int numStates = width * height;
    double featureWeights[] = {-0.01, -0.61, -0.38};
    double gamma = 0.95;
    
    double** stateFeatures = cakmakWorld1b(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {4,
                                      8,9,10,11,12,
                                      15,19,
                                      22, 26,
                                      29,30,31,32,33,
                                      35,36};
//    vector<unsigned int> initStates = {33};
    vector<unsigned int> termStates = {1};
    vector<unsigned int> wallStates = {0,  2,3,  5,6,
                                       7,          13,
                                       14,  16,17,18,  20,
                                       21,  23,24,25,  27,
                                       28,             34,
                                           37,38,39,40,41};
    bool stochastic = false;
   
    FeatureMDP* world1 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world1)->setWallState(s);
        

    return world1;
}


//return fully formed cakmak world 1 with task 1 weights
FeatureMDP* generateCakmakTask1()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 7;
    const int height = 6;
    const int numStates = width * height;
    double featureWeights[] = {2,-1,-1};
    double gamma = 0.95;
    
    double** stateFeatures = cakmakWorld1(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {4,
                                      8,9,10,11,12,
                                      15,19,
                                      22, 26,
                                      29,30,31,32,33,
                                      35,36};
//    vector<unsigned int> initStates = {33};
    vector<unsigned int> termStates = {1};
    vector<unsigned int> wallStates = {0,  2,3,  5,6,
                                       7,          13,
                                       14,  16,17,18,  20,
                                       21,  23,24,25,  27,
                                       28,             34,
                                           37,38,39,40,41};
    bool stochastic = false;
   
    FeatureMDP* world1 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world1)->setWallState(s);
        

    return world1;
}


//return fully formed cakmak world 1 with task 2 weights
FeatureMDP* generateCakmakTask2()
{
    double featureWeights[] = {2,-1,-10};
    FeatureMDP* world2 = generateCakmakTask1();
    world2->setFeatureWeights(featureWeights);
    return world2;
}

//return fully formed cakmak world 1 with task 1 weights
FeatureMDP* generateCakmakTask3()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 6;
    const int height = 6;
    const int numStates = width * height;
    double featureWeights[] = {1,1,-1};
    double gamma = 0.95;
    
    double** stateFeatures = cakmakWorld2(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {    2,3,
                                       6,  8,9,  11,
                                       12,13,14,15,16,17,
                                       18,19,20,21,22,23,
                                       24,25,26,27,28,29,
                                       30,31,32,33,34,35};
    vector<unsigned int> termStates = {0,5};
    vector<unsigned int> wallStates = {1,4,7,10};
    bool stochastic = false;
   
    FeatureMDP* world2 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world2)->setWallState(s);
        

    return world2;
}



//return fully formed cakmak world 1 with task 1 weights
FeatureMDP* generateCakmakTask3b()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 6;
    const int height = 3;
    const int numStates = width * height;
    double featureWeights[] = {1,1,-1};
    double gamma = 0.95;
    
    double** stateFeatures = cakmakWorld2b(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {    2,3,
                                       6,  8,9,  11,
                                       12,13,14,15,16,17};
    vector<unsigned int> termStates = {0,5};
    vector<unsigned int> wallStates = {1,4,7,10};
    bool stochastic = false;
   
    FeatureMDP* world2 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world2)->setWallState(s);
        

    return world2;
}

//return fully formed cakmak world 1 with task 1 weights
FeatureMDP* generateCakmakTask3c()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 2;
    const int height = 2;
    const int numStates = width * height;
    double featureWeights[] = {1,1,-1};
    double gamma = 0.5;
    
    double** stateFeatures = cakmakWorld2c(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {    2,3};
    vector<unsigned int> termStates = {0,1};
    vector<unsigned int> wallStates;
    bool stochastic = false;
   
    FeatureMDP* world2 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world2)->setWallState(s);
        

    return world2;
}


//return fully formed cakmak world 1 with task 1 weights
FeatureMDP* generateCakmakTask3d()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 4;
    const int height = 2;
    const int numStates = width * height;
    double featureWeights[] = {1,1,-1};
    double gamma = 0.95;
    
    double** stateFeatures = cakmakWorld2d(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {1,2,4,5,6,7};
    vector<unsigned int> termStates = {0,3};
    vector<unsigned int> wallStates;
    bool stochastic = false;
   
    FeatureMDP* world2 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world2)->setWallState(s);
        

    return world2;
}


//simple non-terminal version
FeatureMDP* generateCakmakTask3e()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 4;
    const int height = 2;
    const int numStates = width * height;
    double featureWeights[] = {1,1,-1};
    double gamma = 0.5;
    
    double** stateFeatures = cakmakWorld2d(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {0,1,2,3,4,5,6,7};
    vector<unsigned int> termStates = {};
    vector<unsigned int> wallStates = {};
    bool stochastic = false;
   
    FeatureMDP* world2 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
            ((FeatureGridMDP*) world2)->setWallState(s);
        

    return world2;
}

//return fully formed cakmak world 1 with task 1 weights
FeatureMDP* generateCakmakTask3f()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 4;
    const int height = 2;
    const int numStates = width * height;
    double featureWeights[] = {1,3,-1};
    double gamma = 0.95;
    
    double** stateFeatures = cakmakWorld2d(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {1,2,4,5,6,7};
    vector<unsigned int> termStates = {0,3};
    vector<unsigned int> wallStates;
    bool stochastic = false;
   
    FeatureMDP* world2 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
            ((FeatureGridMDP*) world2)->setWallState(s);
        

    return world2;
}


//Chain to test optimal teaching
FeatureMDP* generateOptimalTeachingChainTest()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 6;
    const int height = 1;
    const int numStates = width * height;
    double featureWeights[] = {1,3,-1};
    double gamma = 0.95;
    
    double** stateFeatures = cakmakWorld2Chain(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {1,2,3,4};
    vector<unsigned int> termStates = {0,5};
    vector<unsigned int> wallStates;
    bool stochastic = false;
   
    FeatureMDP* world = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world)->setWallState(s);
        

    return world;
}


//TODO does it still work if second weight is 2?
FeatureMDP* generateCakmakTask4()
{
    FeatureMDP* task4 = generateCakmakTask3();
    double featureWeights[] = {1,3,0};
    task4->setFeatureWeights(featureWeights);
    task4->setDiscount(0.95);
    //cout << "discount = " << task4->getDiscount();
    
    return task4;
}



FeatureMDP* generateStochasticDebugMDP()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 3;
    const int height = 2;
    const int numStates = width * height;
    double featureWeights[] = {1,1,-1};
    double gamma = 0.95;
    
    double** stateFeatures = stochasticDebugWorld(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {1,3,4,5};
    vector<unsigned int> termStates = {0,2};
    bool stochastic = false;
   
    FeatureMDP* debug = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
        

    return debug;
}

//truncated version of Cakmak Task4 
FeatureMDP* generateStochasticDebugMDP2()
{
    const int numFeatures = 3; //star, grey, darkgrey
    const int width = 6;
    const int height = 3;
    const int numStates = width * height;
    double featureWeights[] = {1,2,-1};
    double gamma = 1.0;
    
    double** stateFeatures = stochasticDebugWorld2(numFeatures, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {    2,3,
                                       6,  8,9,  11,
                                       12,13,14,15,16,17};
    vector<unsigned int> termStates = {0,5};
    vector<unsigned int> wallStates = {1,4,7,10};
    bool stochastic = false;
   
    FeatureMDP* world2 = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world2)->setWallState(s);
        

    return world2;
}


//TODO get this to actually work! It picks state 11 still, why?
//return fully formed cakmak world that says zero uncertainty when there still is uncertainty
FeatureMDP* generateCakmakBadExample1()
{
    const int numFeatures = 5; //star, grey, darkgrey
    const int width = 3;
    const int height = 4;
    const int numStates = width * height;
    double featureWeights[] = {-1,2,-1,-1,-10};
    double gamma = 0.95;
    
    double** stateFeatures = cakmakBadWorld1(numFeatures, width, height, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {1,11};
    vector<unsigned int> termStates = {5};
    vector<unsigned int> wallStates = {};
    bool stochastic = false;
   
    FeatureMDP* world_bad = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world_bad)->setWallState(s);
        

    return world_bad;
}


//return fully formed cakmak world that says zero uncertainty when there still is uncertainty
FeatureMDP* generateCakmakBadExample2()
{
    const int numFeatures = 5; //star, grey, darkgrey
    const int width =7;
    const int height = 8;
    const int numStates = width * height;
    double featureWeights[] = {+2,-1,-1,-10,-1}; //s,g,d,r,y
    double gamma = 0.95;
    
    double** stateFeatures = cakmakBadWorld2(numFeatures, width, height, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {8,9,10,11,12,13,15,18,
                                    22,23,24,25,26,29,33,36,40,43,44,45,
                                    46,47,49,50};
    vector<unsigned int> termStates = {15};
    vector<unsigned int> wallStates = {0,1,2,3,4,5,6,
                                       7,14,16,17,19,20,
                                       21,27,
                                       28,30,31,32,34,
                                       35,37,38,39,41,
                                       42,48,
                                       51,52,53,54,55};
    bool stochastic = false;
   
    FeatureMDP* world_bad = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world_bad)->setWallState(s);
        

    return world_bad;
}


//return fully formed cakmak world that says zero uncertainty when there still is uncertainty
FeatureMDP* generateCakmakBadExample3()
{
    const int numFeatures = 7; //star, grey, darkgrey
    const int width =4;
    const int height = 4;
    const int numStates = width * height;
    double featureWeights[] = {+2,0,0,-10,0,0,-10}; //s,g,d,r,y,b,o
    double gamma = 0.5;
    
    double** stateFeatures = cakmakBadWorld3(numFeatures, width, height, numStates);
    //double** stateFeatures = initFeaturesToyFeatureDomain5x5(numStates, numFeatures);
    //double** stateFeatures = initFeatureCountDebugDomain2x2(numStates, numFeatures);
    
    //set up terminals and inits
    vector<unsigned int> initStates = {2,3,4,5,7,8,9,10,12,13,14};
    vector<unsigned int> termStates = {6};
    vector<unsigned int> wallStates = {0,1,11,15};
    bool stochastic = false;
   
    FeatureMDP* world_bad = new FeatureGridMDP(width, height, initStates, termStates, numFeatures, featureWeights, stateFeatures, stochastic, gamma);
    
    
    //set wall states
    for(unsigned int s : wallStates)
        ((FeatureGridMDP*) world_bad)->setWallState(s);
        

    return world_bad;
}


map<pair<unsigned int,unsigned int>, vector<double> > calculateStateActionFCounts(vector<unsigned int> & opt_policy, FeatureMDP* fmdp, double eps)
{
    unsigned int numActions = fmdp->getNumActions();
    unsigned int numStates = fmdp->getNumStates();
    unsigned int numFeatures = fmdp->getNumFeatures();
    double gamma = fmdp->getDiscount();
    double*** T = fmdp->getTransitions();
    
    //calculate state_fcounts
    double** stateFcounts = calculateStateExpectedFeatureCounts(opt_policy, fmdp, eps);
    //for each state action pair, calculate the expected feature counts following this action from this state and add to the map
    map<pair<unsigned int,unsigned int>, vector<double> > fcount_map; 
   
    for(unsigned int s = 0; s < numStates; s++)
    {
        for(unsigned int a = 0; a < numActions; a++)
        {
            //store fcounts for a particular sa pair
            //initialize to feature for that state
            vector<double> counts(fmdp->getStateFeature(s), fmdp->getStateFeature(s) + numFeatures);
            //calculate by adding features for s with expected fcounts after taking action a
            for(unsigned int s2 = 0; s2 < numStates; s2++)
            {
                if(T[s][a][s2] > 0)
                {       
                    //cout << "adding transition to state: " << s2 << endl;
                    //accumulate features for state s2
                    for(unsigned int f = 0; f < numFeatures; f++)
                        counts[f] += gamma * T[s][a][s2] * stateFcounts[s2][f];
                }
            }
            
            //add to the map
            fcount_map[make_pair(s,a)] = counts; 
        }
    }
            
   
    //clean up
    for(unsigned int s = 0; s < fmdp->getNumStates(); s++)
        delete[] stateFcounts[s];
    delete[] stateFcounts;  
    
    
    return fcount_map;

}


map<pair<unsigned int,unsigned int>, vector<double> > calculateStateActionFCounts(vector<vector<double> > & opt_policy, FeatureMDP* fmdp, double eps)
{
    unsigned int numActions = fmdp->getNumActions();
    unsigned int numStates = fmdp->getNumStates();
    unsigned int numFeatures = fmdp->getNumFeatures();
    double gamma = fmdp->getDiscount();
    double*** T = fmdp->getTransitions();
    
    //calculate state_fcounts
    double** stateFcounts = calculateStateExpectedFeatureCounts(opt_policy, fmdp, eps);
    //for each state action pair, calculate the expected feature counts following this action from this state and add to the map
    map<pair<unsigned int,unsigned int>, vector<double> > fcount_map; 
   
    for(unsigned int s = 0; s < numStates; s++)
    {
        for(unsigned int a = 0; a < numActions; a++)
        {
            //store fcounts for a particular sa pair
            //initialize to feature for that state
            vector<double> counts(fmdp->getStateFeature(s), fmdp->getStateFeature(s) + numFeatures);
            //calculate by adding features for s with expected fcounts after taking action a
            for(unsigned int s2 = 0; s2 < numStates; s2++)
            {
                if(T[s][a][s2] > 0)
                {       
                    //cout << "adding transition to state: " << s2 << endl;
                    //accumulate features for state s2
                    for(unsigned int f = 0; f < numFeatures; f++)
                        counts[f] += gamma * T[s][a][s2] * stateFcounts[s2][f];
                }
            }
            
            //add to the map
            fcount_map[make_pair(s,a)] = counts; 
        }
    }
            
   
    //clean up
    for(unsigned int s = 0; s < fmdp->getNumStates(); s++)
        delete[] stateFcounts[s];
    delete[] stateFcounts;  
    
    
    return fcount_map;

}


//perform dot product  w^T (mu_sa - mu_sb)
double w_dot_mu_diff(double* w, vector<double> mu_sa, vector<double> mu_sb, double eps)
{
    assert(mu_sa.size() == mu_sb.size());
    //check to see if the constraint is zero within eps
    bool all_zero = true;
    for(unsigned int i = 0; i < mu_sa.size(); i++)
        if(abs(mu_sa[i] - mu_sb[i]) > eps)
            all_zero = false;
    
    if(all_zero)
        return 0.0;
    
    double dot_prod = 0;

    for(unsigned int i = 0; i < mu_sa.size(); i++)
        dot_prod += w[i] * (mu_sa[i] - mu_sb[i]);
    return dot_prod;

}

bool coneContainsPoint(vector<pair<unsigned int,unsigned int> > D, FeatureMDP* fmdp, vector<pair<unsigned int, unsigned int>> traj, map<pair<unsigned int,unsigned int>, vector<double> >sa_fcounts, double* samp, double eps)
{
    //for each state, calculate the half-spaces for each state-action in trajectory and in D

    unsigned int numActions = fmdp->getNumActions();
    
    //check the trajectory first 
    for(pair<unsigned int, unsigned int> sa : traj)
    {
        //get mu_sa 
        vector<double> mu_sa = sa_fcounts.at(sa);
        //create half-space for all actions
        for(unsigned int a = 0; a < numActions; a++)
        {
            //get mu_sb
            vector<double> mu_sb = sa_fcounts.at(make_pair(sa.first, a));
            //check if sample is outside halfspace
            if(w_dot_mu_diff(samp, mu_sa, mu_sb, eps) < 0.0)
                return false;
            
        }
    }
    
    //then check all elements already in demonstration set D
    for(pair<unsigned int, unsigned int> sa : D)
    {
        //get mu_sa 
        vector<double> mu_sa = sa_fcounts.at(sa);
        //create half-space for all actions
        for(unsigned int a = 0; a < numActions; a++)
        {
            //get mu_sb
            vector<double> mu_sb = sa_fcounts.at(make_pair(sa.first, a));
            //check if sample is outside halfspace
            if(w_dot_mu_diff(samp, mu_sa, mu_sb, eps) < 0.0)
                return false;
            
        }
    }
    
    //if it passed all the tests, then it is inside the convex cone
    return true;


}


double calculateUncertaintyVolume(vector<pair<unsigned int,unsigned int> > D, FeatureMDP* fmdp, vector<pair<unsigned int,unsigned int> > traj, map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts, int numSamples, double eps)
{
    int volCount = 0;
    //get number of features
    int numFeatures = sa_fcounts.at(make_pair(0,0)).size();
    for(int i = 0; i < numSamples; i++)
    {
        //generate sample from hypercube [-1,1]^m
        double samp[numFeatures];
        for(int s=0; s < numFeatures; s++)
            samp[s] = 2 * (rand() / (double) RAND_MAX) - 1;
        
//        for(int s = 0; s < numFeatures; s++)
//            cout << samp[s] << " ";
//        cout << endl;
        
        //test to see if sample inside demo convex cone
        if(coneContainsPoint(D, fmdp, traj, sa_fcounts, samp, eps))
            volCount++;
    }
    
    double volume_est = ((double) volCount) / numSamples;
    return volume_est;

}

MatrixXd convertToMatrixConstraints(vector<pair<unsigned int,unsigned int> > traj, map<pair<unsigned int,unsigned int>, 
                                                                                                vector<double> > sa_fcounts, double eps, FeatureMDP* fmdp)
{
    unsigned int numActions = fmdp->getNumActions();
    unsigned int numFeatures = fmdp->getNumFeatures();
    vector<vector<double> > constraints;  //use this to hold constraints as they are generated
    //get state action fcounts from traj
    for(pair<unsigned int, unsigned int> sa : traj)
    {
        //get mu_sa 
        vector<double> mu_sa = sa_fcounts.at(sa);
        //create half-space for all actions
        for(unsigned int a = 0; a < numActions; a++)
        {
            if(a != sa.second) //make sure actions are different
            {
                //get mu_sb
                vector<double> mu_sb = sa_fcounts.at(make_pair(sa.first, a));
                //check if sample is outside halfspace
                //check to see if the constraint is zero within eps
                bool all_zero = true;
                for(unsigned int i = 0; i < mu_sa.size(); i++)
                    if(abs(mu_sa[i] - mu_sb[i]) > eps)
                        all_zero = false;
                
                if(all_zero)
                    continue;  //skip this feature count diff as it is unconstraining
                
                vector<double> mu_diff;

                for(unsigned int i = 0; i < mu_sa.size(); i++)
                    mu_diff.push_back(mu_sa[i] - mu_sb[i]);
                constraints.push_back(mu_diff);
                    
            }
            
        }
    }
    MatrixXd traj_mat(constraints.size(),numFeatures); //put constraints here once we know how many there are 
    for(int i = 0; i < traj_mat.rows(); i++)
        for(int j=0; j < traj_mat.cols(); j++)
            traj_mat(i,j) = constraints[i][j];
    return traj_mat;

}

void printHalfSpaceConstraints(vector<pair<unsigned int,unsigned int> > D, map<pair<unsigned int,unsigned int>, vector<double> >sa_fcounts, unsigned int numActions)
{
    //for each state, calculate the half-spaces for each state-action in trajectory and in D
    cout <<" **** Half-spaces ****" << endl;
    
    
    //then check all elements already in demonstration set D
    for(pair<unsigned int, unsigned int> sa : D)
    {
        //get mu_sa 
        vector<double> mu_sa = sa_fcounts.at(sa);
        //create half-space for all actions
        for(unsigned int a = 0; a < numActions; a++)
        {
            if(a != sa.second)
            {
                vector<double> mu_sb = sa_fcounts.at(make_pair(sa.first, a));
                //cout << "(" << sa.first << "," << sa.second << ") vs ("  << sa.first << "," << a << ")" << endl;  
                for(unsigned int f=0; f<mu_sb.size(); f++)
                    cout << mu_sa[f] - mu_sb[f] << ", ";
                cout << endl;
            }
        }
    }
    

}
//TODO figure out why I'm not removing all zero constraints, maybe that's just the problem...
//return the demonstration trajectories 
vector<vector<pair<unsigned int,unsigned int> > > OptimalTeachingIRL_Stochastic(FeatureMDP* fmdp, int K, int horizon, int numSamples, double threshold)
{
    double eps = 0.0001;
    unsigned int numStates = fmdp->getNumStates();
    //unsigned int numFeatures = fmdp->getNumFeatures();
    
    //store demonstrations without duplicates
    vector<pair<unsigned int,unsigned int> > D;
    //store full trajectories for teaching
    vector<vector<pair<unsigned int,unsigned int> > > trajs;
    
    ////compute optimal policy based on fdmp
    
    fmdp->valueIteration(eps);
    fmdp->calculateQValues();
    vector< vector<double> > opt_policy = fmdp->getOptimalStochasticPolicy();
//    cout << "-- reward function --" << endl;
//    fmdp->displayRewards();
//    cout << "-- value function ==" << endl;
//    fmdp->displayValues();
//    cout << "-- optimal policy --" << endl;
//    for(unsigned int s=0; s < fmdp->getNumStates(); s++)
//    {
//        if(!fmdp->isWallState(s))
//        {
//            cout << "state " << s << ": ";
//            for(unsigned int a = 0; a < fmdp->getNumActions(); a++)
//                cout << opt_policy[s][a] << ",";
//            cout << endl;
//        }
//    }
    //cout << "-- transitions --" << endl;
    //fmdp->displayTransitions();
    
    ////precompute all the expected feature counts
    map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts = calculateStateActionFCounts(opt_policy, fmdp, eps); 
    
    //print out all of the state action pairs and their expected feature counts
//    for(const auto &myPair : sa_fcounts)
//    {
//        unsigned int s0 = myPair.first.first;
//        if(fmdp->isInitialState(s0))
//        {
//            cout << "(" << myPair.first.first  << "," << myPair.first.second << ")" << endl;
//            for(unsigned int i=0; i<numFeatures; i++)
//                cout << myPair.second[i] << ", ";
//            cout << endl;
//        }
//    }
    

    ////find next best start state and demonstration
    double minVolume = 1000000;
    while(minVolume > threshold)
    {
        vector<pair<unsigned int, unsigned int> > bestTraj;
        ////generate K trajectories from each starting state
        for(unsigned int s0 = 0; s0 < numStates; s0++)
        {
            
            //check if initial state
            if(fmdp->isInitialState(s0))
            {
                cout << "checking initial state " << s0 << endl;
                for(int k = 0; k < K; k++)
                {
                    //generate a sample trajectory following stochastic policy
                    vector<pair<unsigned int, unsigned int>> traj = fmdp->monte_carlo_argmax_rollout(s0, horizon);
                    //print out demos
                    for(pair<unsigned int, unsigned int> p : traj)
                        cout << "(" <<  p.first << "," << p.second << "), ";
                    cout << endl;
                    
                    
                    //skip if already in demonstration set D
                    bool newInfo = false;
                    for(pair<unsigned int, unsigned int> p : traj)
                    {
                        if(std::find(D.begin(), D.end(), p) == D.end())
                            newInfo = true;
                    }
                    if(!newInfo)
                        continue;
                    
                    
                    //compute resulting uncertainty area
                    double newVolume = calculateUncertaintyVolume(D, fmdp, traj, sa_fcounts, numSamples, eps);
                    cout << "new volume = " << newVolume << endl;
                    if(newVolume < minVolume)
                    {
                        minVolume = newVolume;
                        bestTraj = traj;
                    }
                }
            } 
        }
        //what to do if we can't reduce uncertainty any more? (i.e. no best)
        if(bestTraj.size() > 0)
        {
            cout << "Best start state = " << bestTraj[0].first << endl;
            cout << "uncertainty volume = " << minVolume << endl;
        
        
            // add best trajectory to trajs 
            trajs.push_back(bestTraj);
            //add new, non-duplicate state-action pairs to demonstration too
            for(pair<unsigned int, unsigned int> p : bestTraj)
            {
                if(!fmdp->isTerminalState(p.first))
                    if(std::find(D.begin(), D.end(), p) == D.end())
                        D.push_back(p);
            }
        }
        else // can't improve any more
        {
            cout << "Adding more demonstrations will not reduce uncertainty" << endl;
            break;
        }
    }
    cout << "half space constraints" << endl;
    printHalfSpaceConstraints(D, sa_fcounts, fmdp->getNumActions());
    
    return trajs;
}


//This uses Eigen and vectorizes the volume estimation calculation
int countPointsContainedInCone(MatrixXd D_mat, MatrixXd traj_mat, int num_samples, MatrixXd& R)
{
    assert(D_mat.cols() == traj_mat.cols());
    //int n_features = D_mat.cols();
    //create constraint matrix with existing demos and candidate demo trajectory
    MatrixXd C(D_mat.rows()+traj_mat.rows(), D_mat.cols());
    int n_constraints = C.rows();
    C << D_mat, traj_mat;     
    //cout << C.rows() << "x" << C.cols() << endl;
    //MatrixXd R = MatrixXd::Random(n_features, num_samples);
    //std::cout << R << std::endl;
    MatrixXd H = C * R;
    ArrayXd counts = (H.array() >= 0.0).cast<double>().colwise().sum();
    double inside_cnt = (counts == n_constraints).cast<double>().sum();
    return inside_cnt;
    
}


//TODO: get this to use matrices and Eigen library
//return the demonstration trajectories 
vector<vector<pair<unsigned int,unsigned int> > > OptimalTeachingIRL_Cakmak_Vectorized(FeatureMDP* fmdp, int K, int horizon, int numSamples, double threshold, double eps, bool stochastic)
{

    unsigned int numStates = fmdp->getNumStates();
    unsigned int numFeatures = fmdp->getNumFeatures();
    
    //store demonstrations without duplicates
    vector<pair<unsigned int,unsigned int> > D;
    //store full trajectories for teaching
    vector<vector<pair<unsigned int,unsigned int> > > trajs;
    
    ////compute optimal policy based on fdmp
    
    fmdp->valueIteration(eps);
    fmdp->calculateQValues();
    vector<unsigned int> det_pi (fmdp->getNumStates());
    fmdp->getOptimalPolicy(det_pi);
    vector< vector<double> > opt_policy = fmdp->getOptimalStochasticPolicy();
//    cout << "-- reward function --" << endl;
//    fmdp->displayRewards();
//    cout << "-- value function ==" << endl;
//    fmdp->displayValues();
//    cout << "-- optimal policy --" << endl;
//    for(unsigned int s=0; s < fmdp->getNumStates(); s++)
//    {
//        if(!fmdp->isWallState(s))
//        {
//            cout << "state " << s << ": ";
//            for(unsigned int a = 0; a < fmdp->getNumActions(); a++)
//                cout << opt_policy[s][a] << ",";
//            cout << endl;
//        }
//    }
    //cout << "-- transitions --" << endl;
    //fmdp->displayTransitions();
    
    ////precompute all the expected feature counts
    map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts;
    if(stochastic) 
        sa_fcounts = calculateStateActionFCounts(opt_policy, fmdp, eps); 
    else
        sa_fcounts = calculateStateActionFCounts(det_pi, fmdp, eps);
    
    //print out all of the state action pairs and their expected feature counts
//    for(const auto &myPair : sa_fcounts)
//    {
//        unsigned int s0 = myPair.first.first;
//        if(fmdp->isInitialState(s0))
//        {
//            cout << "(" << myPair.first.first  << "," << myPair.first.second << ")" << endl;
//            for(unsigned int i=0; i<numFeatures; i++)
//                cout << myPair.second[i] << ", ";
//            cout << endl;
//        }
//    }
    

    
    ////find next best start state and demonstration
    //I'm converting volumes into plain counts
    int minCount_old = 0;
    int minCount = numSamples;
    int minCount_diff = numSamples;
    MatrixXd R = MatrixXd::Random(numFeatures, numSamples);
    while(minCount_diff > (threshold * numSamples))
    {
        minCount_old = minCount;
        //use matrix to keep track of constraints
        MatrixXd D_mat = convertToMatrixConstraints(D, sa_fcounts, eps, fmdp);


 
        vector<pair<unsigned int, unsigned int> > bestTraj;
        ////generate K trajectories from each starting state
        for(unsigned int s0 = 0; s0 < numStates; s0++)
        {
            
            //check if initial state
            if(fmdp->isInitialState(s0))
            {
                cout << "checking initial state " << s0 << endl;
                for(int k = 0; k < K; k++)
                {
                    vector<pair<unsigned int, unsigned int>> traj;
                    if(stochastic)
                    {
                        //generate a sample trajectory following stochastic policy
                        traj = fmdp->monte_carlo_argmax_rollout(s0, horizon);
                    }
                    else
                    {
                        traj = fmdp->policy_rollout(s0, horizon, det_pi);
                    }
                    //print out demos
                    //for(pair<unsigned int, unsigned int> p : traj)
                    //   cout << "(" <<  p.first << "," << p.second << "), ";
                    //cout << endl;
                    
                    
                    //skip if already in demonstration set D
                    bool newInfo = false;
                    for(pair<unsigned int, unsigned int> p : traj)
                    {
                        if(std::find(D.begin(), D.end(), p) == D.end())
                            newInfo = true;
                    }
                    if(!newInfo)
                        continue;
                    
                    
                    //compute resulting uncertainty area
                    //double newVolume = calculateUncertaintyVolume(D, fmdp, traj, sa_fcounts, numSamples, eps);
                    MatrixXd traj_mat = convertToMatrixConstraints(traj, sa_fcounts, eps, fmdp);
                    int newCount = countPointsContainedInCone(D_mat, traj_mat, numSamples, R);
                    //cout << "new count = " << newCount << endl;
                    if(newCount < minCount)
                    {
                        minCount = newCount;
                        bestTraj = traj;
                       
                    }
                }
            } 
        }
        //what to do if we can't reduce uncertainty any more? (i.e. no best)
        if(bestTraj.size() > 0)
        {
            cout << "Best start state = " << bestTraj[0].first << endl;
            cout << "uncertainty count = " << minCount << endl;
        
        
            // add best trajectory to trajs 
            trajs.push_back(bestTraj);
            //add new, non-duplicate state-action pairs to demonstration too
            for(pair<unsigned int, unsigned int> p : bestTraj)
            {
                if(!fmdp->isTerminalState(p.first))
                    if(std::find(D.begin(), D.end(), p) == D.end())
                        D.push_back(p);
            }

        }
        else // can't improve any more
        {
            cout << "Adding more demonstrations will not reduce uncertainty" << endl;
            break;
        }
        //no need to try again
        if(minCount == 0)
        {
            cout << "Adding more demonstrations will not reduce uncertainty" << endl;
            break;
        }
        minCount_diff = minCount_old - minCount;
    }
    cout << "half space constraints" << endl;
    printHalfSpaceConstraints(D, sa_fcounts, fmdp->getNumActions());
    
    return trajs;
}

//TODO: test this, maybe vectorize it?
// MatrixXd randomSamplesFromCone(MatrixXd C, int numSamples)
// {

//     int n_features = C.cols();
//     int n_constraints = C.rows();
//     //create sampling matrix S
//     MatrixXd S(n_features, numSamples);
//     //cout << C.rows() << "x" << C.cols() << endl;
//     int samplesGenerated = 0;
//     while(samplesGenerated < numSamples)
//     {
//         MatrixXd R;
//         do {
//             R = MatrixXd::Random(n_features, 1);
//             MatrixXd H = C * R;
//             double count = (H.array() >= 0.0).cast<double>().sum();
//         }while(count < n_constraints);
//         S.col(samplesGenerated) = R;
//         samplesGenerated++;
//     }
//     return S;

// }

//calculate and return L2-norm of vec
double L2_norm(vector<double> vec)
{
    double sq_length = 0;
    for(double el : vec)
        sq_length += el * el;
    return sqrt(sq_length); 
}

//make sure each vector<double> has L2-norm of 1.
vector<vector<double> > normalize(vector<vector<double> > constraints)
{
    vector<vector<double> > normalized;
    //zero vector for checking
    vector<double> zeros(constraints[0].size());
    for(unsigned int i=0;i<zeros.size();i++) zeros[i] = 0;
    for(vector<double> c : constraints)
    {
        if(areEqual(c,zeros))
            continue;
        vector<double> c_normed;
        double length = L2_norm(c);
        for(double el : c)
            c_normed.push_back(el / length);
        normalized.push_back(c_normed);
    }
    return normalized;
}

//not sure what this should return if a or b is nan...
bool areEqual(vector<double> a, vector<double> b)
{
    for(double a_i : a)
        assert(!std::isnan(a_i));
    for(double b_i : b)
        assert(!std::isnan(b_i));
    if(a.size() != b.size())
        return false;
    for(unsigned int i=0; i < a.size(); i++)
        if(abs(a[i] - b[i]) > 0.00001)
            return false;
    return true;
}

//remove vector<doubles> that are duplicates
vector<vector<double> > removeDuplicates(vector<vector<double> > constraints)
{
    //debugging, print out starting constraints
//    cout << "in removeDuplicates()" <<endl;
//    cout << "initial constraints" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double d : constr)
//            cout << d << ",";
//        cout << endl;
//    }
    ///end debugging
    vector<vector<double> > no_dups;
    for(vector<double> c : constraints)
    {
        //add to no_dups if not already there
        bool addIt = true;
        for(vector<double> d : no_dups)
        {
            if(areEqual(c,d))
            {
                addIt = false;
                break;
            }
        }
        if(addIt)
            no_dups.push_back(c);      
        
        
    }
    //debugging print out ending constraints
//    cout << "final constraints" << endl;
//    for(vector<double> constr : no_dups)
//    {
//        for(double d : constr)
//            cout << d << ",";
//        cout << endl;
//    }
    ///end debugging

    return no_dups;
}

vector<vector<pair<unsigned int, unsigned int> > > generateCandidateTrajectories(vector<vector<double> > opt_policy, FeatureMDP* fmdp, int K, int horizon)
{
    vector<vector<pair<unsigned int, unsigned int> > > trajs;
    for(unsigned int s0 = 0; s0 < fmdp->getNumStates(); s0++)
    {
        //check if initial state
        if(fmdp->isInitialState(s0))
        {
            //cout << "generating from initial state " << s0 << endl;
            for(int k = 0; k < K; k++)
            {
                //generate a sample trajectory following stochastic policy
                vector<pair<unsigned int, unsigned int>> traj = fmdp->monte_carlo_argmax_rollout(s0, horizon);
                trajs.push_back(traj);
            }
        }
    }
    return trajs;
    
}



vector<vector<pair<unsigned int, unsigned int> > > generateCandidateTrajectories(vector<unsigned int> det_policy, FeatureMDP* fmdp, int horizon)
{
    vector<vector<pair<unsigned int, unsigned int> > > trajs;
    for(unsigned int s0 = 0; s0 < fmdp->getNumStates(); s0++)
    {
        //check if initial state
        if(fmdp->isInitialState(s0))
        {
            
                //generate a sample trajectory following stochastic policy
                vector<pair<unsigned int, unsigned int>> traj = fmdp->policy_rollout(s0, horizon, det_policy);
                trajs.push_back(traj);
            
        }
    }
    return trajs;
    
}


////NOTE: this doesn't remove duplicates!!!
//TODO can we do this for individual state-action pairs or does this type of analysis for optimal teaching only work for trajectories
vector<vector<double> > getAllConstraintsForTraj(vector<pair<unsigned int, unsigned int> > traj, map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts, FeatureMDP* fmdp)
{
    //get constraints for half-spaces formed by entire trajectory
    vector<vector<double> > constraints;
    for(pair<unsigned int, unsigned int> p : traj)
    {
        unsigned int s = p.first;
        unsigned int a = p.second;
        if(!fmdp->isTerminalState(s))
        {
            vector<double> mu_sa = sa_fcounts.at(make_pair(s,a));
            for(unsigned int b = 0; b < fmdp->getNumActions(); b++)
            {
                if(a != b)
                {
                    vector<double> mu_sb = sa_fcounts.at(make_pair(s, b));
                    //cout << "(" << s << "," << a << ") vs (" << s << "," << b << ")" << endl; 
                    vector<double> diff;
                    for(unsigned int f=0; f<mu_sa.size();f++)
                    {
                        //cout << mu_sa[f] - mu_sb[f] << endl;
                        diff.push_back(mu_sa[f] - mu_sb[f]);
                    }                    
                    constraints.push_back(diff);
                }
            
            }
        }
    }
    constraints = removeDuplicates(normalize(constraints));  
    return constraints;

}

vector<vector<double> > getDemonstrationFeasibleRegion(vector<pair<unsigned int, unsigned int> > demo, map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts, FeatureMDP* fmdp)
{
    //sort demo in lexicographic order based on states and then actions
    std::sort (demo.begin(), demo.end(), state_action_comp);
    //cout << "sorted demos" << endl;
    //for(pair<unsigned int, unsigned int> p : demo)
    //    cout << "(" << p.first <<"," << p.second << ") ";
    //cout << endl;

    //get constraints for half-spaces formed by entire trajectory
    vector<vector<double> > constraints;
    for(pair<unsigned int, unsigned int> p : demo)
    {
        unsigned int s = p.first;
        unsigned int a = p.second;
        if(!fmdp->isTerminalState(s))
        {
            vector<double> mu_sa = sa_fcounts.at(make_pair(s,a));
            for(unsigned int b = 0; b < fmdp->getNumActions(); b++)
            {
                if(a != b)
                {
                    vector<double> mu_sb = sa_fcounts.at(make_pair(s, b));
                    //cout << "(" << s << "," << a << ") vs (" << s << "," << b << ")" << endl; 
                    vector<double> diff;
                    for(unsigned int f=0; f<mu_sa.size();f++)
                    {
                        //cout << mu_sa[f] - mu_sb[f] << endl;
                        diff.push_back(mu_sa[f] - mu_sb[f]);
                    }                    
                    constraints.push_back(diff);
                }
            
            }
        }
    }
    constraints = removeDuplicates(normalize(constraints));  
    if(constraints.size() > 1)
        return removeRedundantHalfspaces(constraints);
    else
        return constraints;

}



int countNewCovers(vector<vector<double> > constraints_new, vector<vector<double> > constraint_set, vector<bool> covered)
{
    //go through each element of constraints_new and see if it matches an uncovered element of constraint_set
    int count = 0;
    for(vector<double> c_new : constraints_new)
    {
        for(unsigned int i = 0; i < constraint_set.size(); i++)
        {   
            if(areEqual(c_new, constraint_set[i]) && !covered[i])
                count++;
        }
    }
    return count;

}

//TODO use better data structures!
//Assumes that constraints_new has no duplicates!!
int countCovered(vector<vector<double> > constraints_new, vector<vector<double> > constraint_set)
{
    //go through each element of constraints_new and see if it matches an element of constraint_set
    int count = 0;
    for(vector<double> c_new : constraints_new)
    {
        for(unsigned int i = 0; i < constraint_set.size(); i++)
        {   
            if(areEqual(c_new, constraint_set[i]))
                count++;
        }
    }
    return count;

}


vector<bool> updateCoveredConstraints(vector<vector<double> > constraints_added, vector<vector<double> > constraint_set, vector<bool> covered)
{
    for(vector<double> c_new : constraints_added)
    {
        for(unsigned int i = 0; i < constraint_set.size(); i++)
        {   
            if(areEqual(c_new, constraint_set[i]) && !covered[i])
                covered[i] = true;
        }
    }
    return covered;
}


//get feasible region constraints
vector<vector<double> >  getFeasibleRegion(FeatureMDP* fmdp)
{
    double eps = 0.0001;
    unsigned int numStates = fmdp->getNumStates();
    unsigned int numActions = fmdp->getNumActions();
    //unsigned int numFeatures = fmdp->getNumFeatures();
    
    //store demonstrations without duplicates
    vector<vector<pair<unsigned int,unsigned int> > > opt_demos;
    
    ////compute optimal policy based on fdmp
    
    fmdp->valueIteration(eps);
    fmdp->calculateQValues();
    vector< vector<double> > opt_policy = fmdp->getOptimalStochasticPolicy();
//    cout << "-- reward function --" << endl;
//    fmdp->displayRewards();
//    cout << "-- value function ==" << endl;
//    fmdp->displayValues();
   // cout << "-- optimal stochastic policy --" << endl;
   // for(unsigned int s=0; s < fmdp->getNumStates(); s++)
   // {
   //     if(!fmdp->isWallState(s))
   //     {
   //         cout << "state " << s << ": ";
   //         for(unsigned int a = 0; a < fmdp->getNumActions(); a++)
   //             cout << opt_policy[s][a] << ",";
   //         cout << endl;
   //     }
   // }
    //cout << "-- transitions --" << endl;
    //fmdp->displayTransitions();
    
    ////precompute all the expected feature counts
    map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts = calculateStateActionFCounts(opt_policy, fmdp, eps); 
    
    //debugging
//    cout << "debugging s-a fcounts" << endl;
//        //print out all of the state action pairs and their expected feature counts
//    for(const auto &myPair : sa_fcounts)
//    {
//        unsigned int s0 = myPair.first.first;
//        if(fmdp->isInitialState(s0))
//        {
//            cout << "(" << myPair.first.first  << "," << myPair.first.second << ")" << endl;
//            for(unsigned int i=0; i<numFeatures; i++)
//                cout << myPair.second[i] << ", ";
//            cout << endl;
//        }
//    }
    
    //get constraints for half-spaces formed by all optimal actions
    vector<vector<double> > constraints;
    for(unsigned int s = 0; s < numStates; s++)
    {
        if(!fmdp->isTerminalState(s) && !fmdp->isWallState(s))
        {
            for(unsigned int a = 0; a < numActions; a++)
            {
                if(fmdp->isOptimalAction(s,a))
                {
                    vector<double> mu_sa = sa_fcounts.at(make_pair(s,a));
                    for(unsigned int b = 0; b < numActions; b++)
                    {
                        if(a != b)
                        {
                            vector<double> mu_sb = sa_fcounts.at(make_pair(s, b));
                            //cout << "(" << s << "," << a << ") vs (" << s << "," << b << ")" << endl; 
                            vector<double> diff;
                            for(unsigned int f=0; f<mu_sa.size();f++)
                                diff.push_back(mu_sa[f] - mu_sb[f]);
                                
                            constraints.push_back(diff);
                        }
                    }
                    
                }
            }
        }
    }
//    cout << "----Unnormalized half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    constraints = normalize(constraints);

//    cout << "----Normalized half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    constraints = removeDuplicates(constraints);  
    
//    cout << "----Unique half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    
    //remove redundant halfspaces
    //cout << "--removing redundancies" << endl;
    vector<vector<double> > constraint_set = removeRedundantHalfspaces(constraints);
    
//    cout << "--Non-redundant half spaces" << endl;
//    for(vector<double> constr : constraint_set)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }
    return constraint_set;
}


//K is number of rollouts per start state and horizon is the length of rollouts
vector<vector<pair<unsigned int, unsigned int > > > solveSetCoverOptimalTeaching(FeatureMDP* fmdp, int K, int horizon, double eps, bool stochastic)
{
    //cout << "horizon " << horizon << endl;
    unsigned int numStates = fmdp->getNumStates();
    unsigned int numActions = fmdp->getNumActions();
    //unsigned int numFeatures = fmdp->getNumFeatures();
    
    //store demonstrations without duplicates
    vector<vector<pair<unsigned int,unsigned int> > > opt_demos;
    
    ////compute optimal policy based on fdmp
    
    fmdp->valueIteration(eps);
    fmdp->calculateQValues();
    vector<unsigned int> det_pi (fmdp->getNumStates());
    fmdp->getOptimalPolicy(det_pi);
    vector< vector<double> > opt_policy = fmdp->getOptimalStochasticPolicy();
    //cout << "-- reward function --" << endl;
    //fmdp->displayRewards();
    //cout << " -- deterministic policy -- " << endl;
    //fmdp->displayPolicy(det_pi);
    // cout << "-- value function ==" << endl;
    // fmdp->displayValues();
    // cout << "-- optimal policy --" << endl;
    // for(unsigned int s=0; s < fmdp->getNumStates(); s++)
    // {
    //     if(!fmdp->isWallState(s))
    //     {
    //         cout << "state " << s << ": ";
    //         for(unsigned int a = 0; a < fmdp->getNumActions(); a++)
    //             cout << opt_policy[s][a] << ",";
    //         cout << endl;
    //     }
    // }
    //cout << "-- transitions --" << endl;
    //fmdp->displayTransitions();
    
    ////precompute all the expected feature counts
    map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts;
    if(stochastic) 
        sa_fcounts = calculateStateActionFCounts(opt_policy, fmdp, eps); 
    else
        sa_fcounts = calculateStateActionFCounts(det_pi, fmdp, eps);
    
    //debugging
//    cout << "debugging s-a fcounts" << endl;
//        //print out all of the state action pairs and their expected feature counts
//    for(const auto &myPair : sa_fcounts)
//    {
//        unsigned int s0 = myPair.first.first;
//        if(fmdp->isInitialState(s0))
//        {
//            cout << "(" << myPair.first.first  << "," << myPair.first.second << ")" << endl;
//            for(unsigned int i=0; i<numFeatures; i++)
//                cout << myPair.second[i] << ", ";
//            cout << endl;
//        }
//    }
    
    //get constraints for half-spaces formed by all optimal actions only on initial states!
    vector<vector<double> > constraints;
    if(stochastic)
    {
        
        for(unsigned int s = 0; s < numStates; s++)
        {
            if(fmdp->isInitialState(s))
            {
                if(!fmdp->isTerminalState(s) && !fmdp->isWallState(s))
                {
                    for(unsigned int a = 0; a < numActions; a++)
                    {
                        if(fmdp->isOptimalAction(s,a))
                        {
                            vector<double> mu_sa = sa_fcounts.at(make_pair(s,a));
                            for(unsigned int b = 0; b < numActions; b++)
                            {
                                if(a != b)
                                {
                                    vector<double> mu_sb = sa_fcounts.at(make_pair(s, b));
                                    //cout << "(" << s << "," << a << ") vs (" << s << "," << b << ")" << endl; 
                                    vector<double> diff;
                                    for(unsigned int f=0; f<mu_sa.size();f++)
                                        diff.push_back(mu_sa[f] - mu_sb[f]);
                                        
                                    constraints.push_back(diff);
                                }
                            }
                            
                        }
                    }
                }
            }
        }
    }
    else
    {
        
        for(unsigned int s = 0; s < numStates; s++)
        {
            if(fmdp->isInitialState(s))
            {
                if(!fmdp->isTerminalState(s) && !fmdp->isWallState(s))
                {
                    unsigned int a = det_pi[s];
                    vector<double> mu_sa = sa_fcounts.at(make_pair(s,a));
                    for(unsigned int b = 0; b < numActions; b++)
                    {
                        if(a != b)
                        {
                            vector<double> mu_sb = sa_fcounts.at(make_pair(s, b));
                            //cout << "(" << s << "," << a << ") vs (" << s << "," << b << ")" << endl; 
                            vector<double> diff;
                            for(unsigned int f=0; f<mu_sa.size();f++)
                                diff.push_back(mu_sa[f] - mu_sb[f]);
                                
                            constraints.push_back(diff);
                        }
                    }
                    
                }
            }
        }

    }
//    cout << "----Unnormalized half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    constraints = normalize(constraints);

//    cout << "----Normalized half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    constraints = removeDuplicates(constraints);  
    
   // cout << "----Unique half spaces" << endl;
   // for(vector<double> constr : constraints)
   // {
   //     for(double c : constr)
   //         cout << c << ",";
   //     cout << endl;

   // }

 
    //remove redundant halfspaces
    //cout << "--removing redundancies" << endl;
    vector<vector<double> > constraint_set = removeRedundantHalfspaces(constraints);
    
    // cout << "--Non-redundant half spaces" << endl;
    // for(vector<double> constr : constraint_set)
    // {
    //     for(double c : constr)
    //         cout << c << ",";
    //     cout << endl;

    // }


    
    //generate k trajectories from each start state, removing duplicates
    vector<vector<pair<unsigned int, unsigned int> > > candTrajs;
    if(stochastic)
        candTrajs = generateCandidateTrajectories(opt_policy, fmdp, K, horizon);
    else
        candTrajs = generateCandidateTrajectories(det_pi, fmdp, horizon);

    //debug: print out candidate trajectories to see if they match what I think
//    cout << "candidate trajectories" << endl;
//    for(vector<pair<unsigned int, unsigned int> > traj : candTrajs)
//    {
//        for(pair<unsigned int, unsigned int> p : traj)
//                cout << "(" << p.first << "," << p.second << "), ";
//            cout << endl;
//    }

    //create boolean bookkeeping to see what has been covered in the set
    vector<bool> covered(constraint_set.size());
    for(unsigned int i=0;i<covered.size();i++)
        covered[i] = false;
        
    //for each candidate demonstration trajectory check how many uncovered set elements it covers and find one with max added covers
    unsigned int total_covered = 0;
    while(total_covered < constraint_set.size())
    {
        int max_count = 0;
        vector<pair<unsigned int, unsigned int> > best_traj;
        vector<vector<double> > constraints_added;
        for(vector<pair<unsigned int, unsigned int> > traj : candTrajs)
        {
            //debugging
//            cout << "trajectory" << endl;
//            for(pair<unsigned int, unsigned int> p : traj)
//                cout << "(" << p.first << "," << p.second << "), ";
//            cout << endl;
            //end debug
            vector<vector<double> > constraints_new = getAllConstraintsForTraj(traj, sa_fcounts, fmdp);
            //debugging
//            cout << "** traj constraints" << endl;
//            for(vector<double> c : constraints_new)
//            {
//                for(double d : c)
//                    cout << d << ",";
//                cout << endl;
//            }
//            cout << "------" << endl;
            //end debug
            int count = countNewCovers(constraints_new, constraint_set, covered);
            //begin debug
            //cout << "count covered = " << count << endl;
            //end debug
            if(count > max_count)
            {
                max_count = count;
                constraints_added = constraints_new;
                best_traj = traj;
            }
        }
        
        //debugging
//        cout << "----------------------_" << endl;
//        cout << "best count = " << max_count << endl;
//        cout << "best trajectory" << endl;
//        for(pair<unsigned int, unsigned int> p : best_traj)
//            cout << "(" <<  p.first << "," << p.second << "), ";
//        cout << endl;
        
        //update covered flags and add best_traj to demo`
        opt_demos.push_back(best_traj);
        covered = updateCoveredConstraints(constraints_added, constraint_set, covered);
        //begin debugging
//        cout << "covered" << endl;
//        for(bool bo : covered)
//            cout << bo << ",";
//            
//        cout << endl;
        //end debugging
        //increment total_covered
        total_covered += max_count;
    }
    return opt_demos;
}

//K is number of rollouts per start state and horizon is the length of rollouts
vector<vector<pair<unsigned int, unsigned int > > > solveSetCoverOptimalTeaching_sol1(FeatureMDP* fmdp, int K, int horizon)
{
    double eps = 0.0001;
    unsigned int numStates = fmdp->getNumStates();
    unsigned int numActions = fmdp->getNumActions();
    //unsigned int numFeatures = fmdp->getNumFeatures();
    
    //store demonstrations without duplicates
    vector<vector<pair<unsigned int,unsigned int> > > opt_demos;
    
    ////compute optimal policy based on fdmp
    
    fmdp->valueIteration(eps);
    fmdp->calculateQValues();
    vector< vector<double> > opt_policy = fmdp->getOptimalStochasticPolicy();
//    cout << "-- reward function --" << endl;
//    fmdp->displayRewards();
//    cout << "-- value function ==" << endl;
//    fmdp->displayValues();
//    cout << "-- optimal policy --" << endl;
//    for(unsigned int s=0; s < fmdp->getNumStates(); s++)
//    {
//        if(!fmdp->isWallState(s))
//        {
//            cout << "state " << s << ": ";
//            for(unsigned int a = 0; a < fmdp->getNumActions(); a++)
//                cout << opt_policy[s][a] << ",";
//            cout << endl;
//        }
//    }
    //cout << "-- transitions --" << endl;
    //fmdp->displayTransitions();
    
    ////precompute all the expected feature counts
    map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts = calculateStateActionFCounts(opt_policy, fmdp, eps); 
    
    //debugging
//    cout << "debugging s-a fcounts" << endl;
//        //print out all of the state action pairs and their expected feature counts
//    for(const auto &myPair : sa_fcounts)
//    {
//        unsigned int s0 = myPair.first.first;
//        if(fmdp->isInitialState(s0))
//        {
//            cout << "(" << myPair.first.first  << "," << myPair.first.second << ")" << endl;
//            for(unsigned int i=0; i<numFeatures; i++)
//                cout << myPair.second[i] << ", ";
//            cout << endl;
//        }
//    }
    
    //get constraints for half-spaces formed by all optimal actions
    vector<vector<double> > constraints;
    for(unsigned int s = 0; s < numStates; s++)
    {
        if(!fmdp->isTerminalState(s) && !fmdp->isWallState(s))
        {
            for(unsigned int a = 0; a < numActions; a++)
            {
                if(fmdp->isOptimalAction(s,a))
                {
                    vector<double> mu_sa = sa_fcounts.at(make_pair(s,a));
                    for(unsigned int b = 0; b < numActions; b++)
                    {
                        if(a != b)
                        {
                            vector<double> mu_sb = sa_fcounts.at(make_pair(s, b));
                            //cout << "(" << s << "," << a << ") vs (" << s << "," << b << ")" << endl; 
                            vector<double> diff;
                            for(unsigned int f=0; f<mu_sa.size();f++)
                                diff.push_back(mu_sa[f] - mu_sb[f]);
                                
                            constraints.push_back(diff);
                        }
                    }
                    
                }
            }
        }
    }
//    cout << "----Unnormalized half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    constraints = normalize(constraints);

//    cout << "----Normalized half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    constraints = removeDuplicates(constraints);  
    
//    cout << "----Unique half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }
    
    //randomly shuffle them
    // using built-in random generator:
    std::random_shuffle ( constraints.begin(), constraints.end() );

    
    //remove redundant halfspaces
    //cout << "--removing redundancies" << endl;
    vector<vector<double> > constraint_set = removeRedundantHalfspaces(constraints);
    
//    cout << "--Non-redundant half spaces" << endl;
//    for(vector<double> constr : constraint_set)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    
    //generate k trajectories from each start state, removing duplicates
    vector<vector<pair<unsigned int, unsigned int> > > candTrajs = generateCandidateTrajectories(opt_policy, fmdp, K, horizon);

    //create boolean bookkeeping to see what has been covered in the set
    vector<bool> covered(constraint_set.size());
    for(unsigned int i=0;i<covered.size();i++)
        covered[i] = false;
        
    //for each candidate demonstration trajectory check how many uncovered set elements it covers and find one with max added covers
    unsigned int total_covered = 0;
    while(total_covered < constraint_set.size())
    {
        int max_count = 0;
        vector<pair<unsigned int, unsigned int> > best_traj;
        vector<vector<double> > constraints_added;
        
        //randomly shuffle sample demo trajectories
        // using built-in random generator:
        std::random_shuffle ( candTrajs.begin(), candTrajs.end() );
        
        for(vector<pair<unsigned int, unsigned int> > traj : candTrajs)
        {
            //debugging
//            cout << "trajectory" << endl;
//            for(pair<unsigned int, unsigned int> p : traj)
//                cout << "(" << p.first << "," << p.second << "), ";
//            cout << endl;
            //end debug
            vector<vector<double> > constraints_new = getAllConstraintsForTraj(traj, sa_fcounts, fmdp);
            //debugging
//            cout << "** traj constraints" << endl;
//            for(vector<double> c : constraints_new)
//            {
//                for(double d : c)
//                    cout << d << ",";
//                cout << endl;
//            }
//            cout << "------" << endl;
            //end debug
            int count = countNewCovers(constraints_new, constraint_set, covered);
            //cout << "count = " << count << endl;
            if(count > max_count)
            {
                max_count = count;
                constraints_added = constraints_new;
                best_traj = traj;
            }
//            else if(count == max_count)
//            {   
//                //randomly break ties by choosing new demo with max_count with 50% prob
//                if(rand() % 5 == 0)
//                {
//                    max_count = count;
//                    constraints_added = constraints_new;
//                    best_traj = traj;
//                }
//            }
        }
        
        //debugging
//        cout << "----------------------_" << endl;
//        cout << "best count = " << max_count << endl;
//        cout << "best trajectory" << endl;
//        for(pair<unsigned int, unsigned int> p : best_traj)
//            cout << "(" <<  p.first << "," << p.second << "), ";
//        cout << endl;
//        
        //update covered flags and add best_traj to demo`
        opt_demos.push_back(best_traj);
        covered = updateCoveredConstraints(constraints_added, constraint_set, covered);
//        cout << "covered" << endl;
//        for(bool bo : covered)
//            cout << bo << ",";
//            
//        cout << endl;
        //increment total_covered
        total_covered += max_count;
    }
    return opt_demos;
}


//doesn't remove redundant constraints!
//K is number of rollouts per start state and horizon is the length of rollouts
vector<vector<pair<unsigned int, unsigned int > > > solveSetCoverBaseline(FeatureMDP* fmdp, int K, int horizon)
{
    //cout << "horizon " << horizon << endl;
    double eps = 0.0001;
    unsigned int numStates = fmdp->getNumStates();
    unsigned int numActions = fmdp->getNumActions();
    //unsigned int numFeatures = fmdp->getNumFeatures();
    
    //store demonstrations without duplicates
    vector<vector<pair<unsigned int,unsigned int> > > opt_demos;
    
    ////compute optimal policy based on fdmp
    
    fmdp->valueIteration(eps);
    fmdp->calculateQValues();
    vector< vector<double> > opt_policy = fmdp->getOptimalStochasticPolicy();
//    cout << "-- reward function --" << endl;
//    fmdp->displayRewards();
//    cout << "-- value function ==" << endl;
//    fmdp->displayValues();
//    cout << "-- optimal policy --" << endl;
//    for(unsigned int s=0; s < fmdp->getNumStates(); s++)
//    {
//        if(!fmdp->isWallState(s))
//        {
//            cout << "state " << s << ": ";
//            for(unsigned int a = 0; a < fmdp->getNumActions(); a++)
//                cout << opt_policy[s][a] << ",";
//            cout << endl;
//        }
//    }
    //cout << "-- transitions --" << endl;
    //fmdp->displayTransitions();
    
    ////precompute all the expected feature counts
    map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts = calculateStateActionFCounts(opt_policy, fmdp, eps); 
    
    //debugging
//    cout << "debugging s-a fcounts" << endl;
//        //print out all of the state action pairs and their expected feature counts
//    for(const auto &myPair : sa_fcounts)
//    {
//        unsigned int s0 = myPair.first.first;
//        if(fmdp->isInitialState(s0))
//        {
//            cout << "(" << myPair.first.first  << "," << myPair.first.second << ")" << endl;
//            for(unsigned int i=0; i<numFeatures; i++)
//                cout << myPair.second[i] << ", ";
//            cout << endl;
//        }
//    }
    
    //get constraints for half-spaces formed by all optimal actions
    vector<vector<double> > constraints;
    for(unsigned int s = 0; s < numStates; s++)
    {
        if(!fmdp->isTerminalState(s) && !fmdp->isWallState(s))
        {
            for(unsigned int a = 0; a < numActions; a++)
            {
                if(fmdp->isOptimalAction(s,a))
                {
                    vector<double> mu_sa = sa_fcounts.at(make_pair(s,a));
                    for(unsigned int b = 0; b < numActions; b++)
                    {
                        if(a != b)
                        {
                            vector<double> mu_sb = sa_fcounts.at(make_pair(s, b));
                            //cout << "(" << s << "," << a << ") vs (" << s << "," << b << ")" << endl; 
                            vector<double> diff;
                            for(unsigned int f=0; f<mu_sa.size();f++)
                                diff.push_back(mu_sa[f] - mu_sb[f]);
                                
                            constraints.push_back(diff);
                        }
                    }
                    
                }
            }
        }
    }
//    cout << "----Unnormalized half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    constraints = normalize(constraints);

//    cout << "----Normalized half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    constraints = removeDuplicates(constraints);  
    
//    cout << "----Unique half spaces" << endl;
//    for(vector<double> constr : constraints)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    
    //remove redundant halfspaces
    //cout << "--removing redundancies" << endl;
    vector<vector<double> > constraint_set = constraints;//removeRedundantHalfspaces(constraints);
    
//    cout << "--Non-redundant half spaces" << endl;
//    for(vector<double> constr : constraint_set)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    
    //generate k trajectories from each start state, removing duplicates
    vector<vector<pair<unsigned int, unsigned int> > > candTrajs = generateCandidateTrajectories(opt_policy, fmdp, K, horizon);

    //create boolean bookkeeping to see what has been covered in the set
    vector<bool> covered(constraint_set.size());
    for(unsigned int i=0;i<covered.size();i++)
        covered[i] = false;
        
    //for each candidate demonstration trajectory check how many uncovered set elements it covers and find one with max added covers
    unsigned int total_covered = 0;
    while(total_covered < constraint_set.size())
    {
        int max_count = 0;
        vector<pair<unsigned int, unsigned int> > best_traj;
        vector<vector<double> > constraints_added;
        for(vector<pair<unsigned int, unsigned int> > traj : candTrajs)
        {
            //debugging
//            cout << "trajectory" << endl;
//            for(pair<unsigned int, unsigned int> p : traj)
//                cout << "(" << p.first << "," << p.second << "), ";
//            cout << endl;
            //end debug
            vector<vector<double> > constraints_new = getAllConstraintsForTraj(traj, sa_fcounts, fmdp);
            //debugging
//            cout << "** traj constraints" << endl;
//            for(vector<double> c : constraints_new)
//            {
//                for(double d : c)
//                    cout << d << ",";
//                cout << endl;
//            }
//            cout << "------" << endl;
            //end debug
            int count = countNewCovers(constraints_new, constraint_set, covered);
            //cout << "count = " << count << endl;
            if(count > max_count)
            {
                max_count = count;
                constraints_added = constraints_new;
                best_traj = traj;
            }
        }
        
        //debugging
//        cout << "----------------------_" << endl;
//        cout << "best count = " << max_count << endl;
//        cout << "best trajectory" << endl;
//        for(pair<unsigned int, unsigned int> p : best_traj)
//            cout << "(" <<  p.first << "," << p.second << "), ";
//        cout << endl;
//        
        //update covered flags and add best_traj to demo`
        opt_demos.push_back(best_traj);
        covered = updateCoveredConstraints(constraints_added, constraint_set, covered);
//        cout << "covered" << endl;
//        for(bool bo : covered)
//            cout << bo << ",";
//            
//        cout << endl;
        //increment total_covered
        total_covered += max_count;
    }
    return opt_demos;
}

//tries to avoid issue where there are multiple possible non-redundant sets and which ones are picked changes
//the number of demos since some demos will match only the non-redundant constraints in some of the sets
//K is number of rollouts per start state and horizon is the length of rollouts
vector<vector<pair<unsigned int, unsigned int > > > solveSetCoverOptimalTeaching_sol3(FeatureMDP* fmdp, int K, int horizon)
{
    double eps = 0.0001;
    unsigned int numStates = fmdp->getNumStates();
    unsigned int numActions = fmdp->getNumActions();
    //unsigned int numFeatures = fmdp->getNumFeatures();
    
    //store demonstrations without duplicates
    vector<vector<pair<unsigned int,unsigned int> > > opt_demos;
    
    ////compute optimal policy based on fdmp
    
    fmdp->valueIteration(eps);
    fmdp->calculateQValues();
    vector< vector<double> > opt_policy = fmdp->getOptimalStochasticPolicy();
    cout << "-- reward function --" << endl;
    fmdp->displayRewards();
    //cout << "-- value function ==" << endl;
    //fmdp->displayValues();
    cout << "-- optimal policy --" << endl;
    for(unsigned int s=0; s < fmdp->getNumStates(); s++)
    {
        if(!fmdp->isWallState(s))
        {
            cout << "state " << s << ": ";
            for(unsigned int a = 0; a < fmdp->getNumActions(); a++)
                cout << opt_policy[s][a] << ",";
            cout << endl;
        }
    }
    //cout << "-- transitions --" << endl;
    //fmdp->displayTransitions();
    
    ////precompute all the expected feature counts
    map<pair<unsigned int,unsigned int>, vector<double> > sa_fcounts = calculateStateActionFCounts(opt_policy, fmdp, eps); 
    
    //debugging
//    cout << "debugging s-a fcounts" << endl;
//        //print out all of the state action pairs and their expected feature counts
//    for(const auto &myPair : sa_fcounts)
//    {
//        unsigned int s0 = myPair.first.first;
//        if(fmdp->isInitialState(s0))
//        {
//            cout << "(" << myPair.first.first  << "," << myPair.first.second << ")" << endl;
//            for(unsigned int i=0; i<numFeatures; i++)
//                cout << myPair.second[i] << ", ";
//            cout << endl;
//        }
//    }
    
    //get constraints for half-spaces formed by all optimal actions
    vector<vector<double> > constraints;
    for(unsigned int s = 0; s < numStates; s++)
    {
        if(!fmdp->isTerminalState(s) && !fmdp->isWallState(s))
        {
            for(unsigned int a = 0; a < numActions; a++)
            {
                if(fmdp->isOptimalAction(s,a))
                {
                    vector<double> mu_sa = sa_fcounts.at(make_pair(s,a));
                    for(unsigned int b = 0; b < numActions; b++)
                    {
                        if(a != b)
                        {
                            vector<double> mu_sb = sa_fcounts.at(make_pair(s, b));
                            //cout << "(" << s << "," << a << ") vs (" << s << "," << b << ")" << endl; 
                            vector<double> diff;
                            for(unsigned int f=0; f<mu_sa.size();f++)
                                diff.push_back(mu_sa[f] - mu_sb[f]);
                                
                            constraints.push_back(diff);
                        }
                    }
                    
                }
            }
        }
    }
    cout << "----Unnormalized half spaces" << endl;
    for(vector<double> constr : constraints)
    {
        for(double c : constr)
            cout << c << ",";
        cout << endl;

    }

    constraints = normalize(constraints);

    cout << "----Normalized half spaces" << endl;
    for(vector<double> constr : constraints)
    {
        for(double c : constr)
            cout << c << ",";
        cout << endl;

    }

    constraints = removeDuplicates(constraints);  
    
    cout << "----Unique half spaces" << endl;
    for(vector<double> constr : constraints)
    {
        for(double c : constr)
            cout << c << ",";
        cout << endl;

    }

    ////dont' remove them yet! wait and remove based on demos////

//    //remove redundant halfspaces
//    cout << "--removing redundancies" << endl;
//    vector<vector<double> > constraint_set = removeRedundantHalfspaces(constraints);
//    
//    cout << "--Non-redundant half spaces" << endl;
//    for(vector<double> constr : constraint_set)
//    {
//        for(double c : constr)
//            cout << c << ",";
//        cout << endl;

//    }

    
    //generate k trajectories from each start state, removing duplicates
    vector<vector<pair<unsigned int, unsigned int> > > candTrajs = generateCandidateTrajectories(opt_policy, fmdp, K, horizon);

    vector<vector<double> > covered_constraints;
        
    //keep adding demos and removing redundant halfspaces until all have been added or are redundant
    while(constraints.size() > 0)
    {
        ////find max cover demo
        //for each candidate demonstration trajectory check how many uncovered set elements it covers and find one with max added covers
        int max_count = 0;
        vector<pair<unsigned int, unsigned int> > best_traj;
        vector<vector<double> > constraints_added;
        //TODO seems like there should be a fast way to weed out candTrajs that I shouldn't consider...
        for(vector<pair<unsigned int, unsigned int> > traj : candTrajs)
        {
            //debugging
//            cout << "trajectory" << endl;
//            for(pair<unsigned int, unsigned int> p : traj)
//                cout << "(" << p.first << "," << p.second << "), ";
//            cout << endl;
            //end debug
            vector<vector<double> > constraints_new = getAllConstraintsForTraj(traj, sa_fcounts, fmdp);
            //debugging
//            cout << "** traj constraints" << endl;
//            for(vector<double> c : constraints_new)
//            {
//                for(double d : c)
//                    cout << d << ",";
//                cout << endl;
//            }
//            cout << "------" << endl;
            //end debug
            int count = countCovered(constraints_new, constraints);
            //cout << "count = " << count << endl;
            if(count > max_count)
            {
                max_count = count;
                constraints_added = constraints_new;
                best_traj = traj;
            }
        }
        
        ////debugging
        cout << "----------------------_" << endl;
        cout << "best count = " << max_count << endl;
        cout << "best trajectory" << endl;
        for(pair<unsigned int, unsigned int> p : best_traj)
            cout << "(" <<  p.first << "," << p.second << "), ";
        cout << endl;
        cout << "** traj constraints" << endl;
        for(vector<double> c : constraints_added)
        {
            for(double d : c)
                cout << d << ",";
            cout << endl;
        }

        ////end debugging
        
        //update demonstration and keep track of covered constraints
        opt_demos.push_back(best_traj);
        for(vector<double> c_added : constraints_added)
            covered_constraints.push_back(c_added);
            
        
        //TODO: remove redundant constraints contraints_added, might help speed, or hurt...
            
        //remove halfspaces in constraints that are redundant given currently covered constraints
        constraints = removeRedundantHalfspaces(constraints, covered_constraints);
        //begin debugging
        cout << "========" << endl;
        cout << "uncovered constraints left" << endl;
        for(vector<double> c : constraints)
        {
            for(double d : c)
                cout << d << ",";
            cout << endl;
        }
        //end debugging
  
    }
    return opt_demos;
}

//computes the dot product between two unit vectors
//Assumes unit normal vectors!!
double vectorDotProduct(vector<double> x, vector<double> y)
{
    assert(x.size() == y.size());
    double result = 0;
    for(unsigned int i=0; i<x.size(); i++)
        result += x[i] * y[i];
    //truncate to be within range [-1,1]
    if(result < -1.0)
        result = -1.0;
    else if(result > 1.0)
        result = 1.0;
    return result;
}

//TODO run full hungarian algorithm!
//returns unnormalized cumulative similarity
double calculateCumulativeSimilarity(vector<vector<double> > feasibleDemo, vector<vector<double> > feasibleR)
{
    //convert to linked list for easy removal and iteration
    
    list<vector<double> > feasibleRemaining;
    for(vector<double> constr : feasibleR)
        feasibleRemaining.push_back(constr);
    //int sizeFeasibleR = feasibleR.size();
    //cout << "size = " << sizeFeasibleR << endl;

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
            
            //cout << "dot = " << vectorDotProduct(constr, match) << endl;
            //cout << (1.0 < vectorDotProduct(constr, match)) << endl;
            //cout << "angle = " << acos(vectorDotProduct(constr, match)) << endl;
            double similarity =  1.0 - acos(vectorDotProduct(constr, match))/M_PI;
            //cout << "similarity = " << similarity << endl;
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

    return cumSim;

}





#endif
