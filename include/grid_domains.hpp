#ifndef grid_domains_h
#define grid_domains_h
#include <assert.h>
#include <stdlib.h>

double** initFeaturesToyFeatureDomain5x5(int numStates, int numFeatures);
double** randomGridNavDomain(int numStates, int numFeatures);
void displayStateColorFeatures(double** stateFeatures, unsigned int gridCols, unsigned int gridRows, unsigned int numFeatures);
bool featuresAreEqual(double* f1, double* f2, int size);
double** initFeatureCountToyDomain4x4(int numStates, int numFeatures);
double** random9x9GridNavGoalWorld();
double** random9x9GridNavGoalWorld8Features();
double** randomNxNGridNavWorld8Features(int N);
double** enoughToyWorld(int width, int height, int numFeatures, int numStates);

//worlds from Algo Teaching Cakmak Lopes paper
double** cakmakWorld1(int numFeatures, int numStates);
double** cakmakWorld2(int numFeatures, int numStates);



//simple world for testing Cakmak algo
double** cakmakWorld1(int numFeatures, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 3);
    assert(numStates == 6*7);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0};
    double grayFeature[] = {0,1,0};
    double darkgrayFeature[] = {0,0,1};

    char features[] = {'w','s','w','w','g','w','w',
                       'w','g','g','g','g','g','w',
                       'w','d','w','w','w','g','w',
                       'w','d','w','w','w','g','w',
                       'w','g','g','g','g','g','w',
                       'g','g','w','w','w','w','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(darkgrayFeature, darkgrayFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}


//simple world for testing Cakmak algo
double** cakmakWorld1b(int numFeatures, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 3);
    assert(numStates == 6*7);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0};
    double grayFeature[] = {0,1,0};
    double darkgrayFeature[] = {0,0,1};

    char features[] = {'w','s','w','w','g','w','w',
                       'w','g','g','g','g','g','w',
                       'w','d','w','w','w','g','w',
                       'w','d','w','w','w','g','w',
                       'w','d','d','d','d','g','w',
                       'g','g','w','w','w','w','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(darkgrayFeature, darkgrayFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}


//simple world for testing Cakmak algo
double** cakmakWorld2(int numFeatures, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 3);
    assert(numStates == 6*6);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0};
    double diamondFeature[] = {0,1,0};
    double grayFeature[] = {0,0,1};
    

    char features[] = {'s','w','g','g','w','d',
                       'g','w','g','g','w','g',
                       'g','g','g','g','g','g',
                       'g','g','g','g','g','g',
                       'g','g','g','g','g','g',
                       'g','g','g','g','g','g'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(diamondFeature, diamondFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}


//smaller Cakmak world 2
double** cakmakWorld2b(int numFeatures, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 3);
    assert(numStates == 6*3);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0};
    double diamondFeature[] = {0,1,0};
    double grayFeature[] = {0,0,1};
    

    char features[] = {'s','w','g','g','w','d',
                       'g','w','g','g','w','g',
                       'g','g','g','g','g','g'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(diamondFeature, diamondFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}

//4 state Cakmak world 2 for debugging
//This one works with set-cover problem since it has a deterministic optimal policy
double** cakmakWorld2c(int numFeatures, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 3);
    assert(numStates == 2*2);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0};
    double diamondFeature[] = {0,1,0};
    double grayFeature[] = {0,0,1};
    

    char features[] = {'s','d',
                       'g','g'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(diamondFeature, diamondFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}

//8 state Cakmak world 2 for debugging
//not sure if this one will work
double** cakmakWorld2d(int numFeatures, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 3);
    assert(numStates == 2*4);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0};
    double diamondFeature[] = {0,1,0};
    double grayFeature[] = {0,0,1};
    

    char features[] = {'s','g','g','d',
                       'g','g','g','g'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(diamondFeature, diamondFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}

//6 state Cakmak world 2 for debugging
//not sure if this one will work
double** cakmakWorld2Chain(int numFeatures, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 3);
    assert(numStates == 6);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0};
    double diamondFeature[] = {0,1,0};
    double grayFeature[] = {0,0,1};
    

    char features[] = {'s','g','g','g','g','d'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(diamondFeature, diamondFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}


//simple world for testing Cakmak algo
double** stochasticDebugWorld(int numFeatures, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 3);
    assert(numStates == 2*3);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0};
    double diamondFeature[] = {0,1,0};
    double grayFeature[] = {0,0,1};
    

    char features[] = {'s','g','d',
                       'g','g','g'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(diamondFeature, diamondFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}

//simple world for testing Cakmak algo
double** stochasticDebugWorld2(int numFeatures, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 3);
    assert(numStates == 3*6);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0};
    double diamondFeature[] = {0,1,0};
    double grayFeature[] = {0,0,1};
    

    char features[] = {'s','w','g','g','w','d',
                       'g','w','g','g','w','g',
                       'g','g','g','g','g','g'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(diamondFeature, diamondFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}





//simple counter-example for Cakmak algo
double** cakmakBadWorld1(int numFeatures, int width, int height, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 5);
    assert(width == 3);
    assert(height == 4);
    assert(numStates == width*height);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {0,1,0,0,0};
    double blackFeature[] = {0,0,1,0,0};
    double grayFeature[] = {0,0,0,1,0};
    double redFeature[] = {0,0,0,0,1};
    

    char features[] = {'b','w','g',
                       'w','w','s',
                       'w','w','r',
                       'w','w','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'b':
                std::copy(blackFeature, blackFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;

        }
    
    }
    return stateFeatures;


}


//simple world for testing Cakmak algo
double** cakmakBadWorld2(int numFeatures, int width, int height, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 5);
    assert(height == 8);
    assert(width == 7);
    assert(numStates == width * height);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0,0,0};
    double grayFeature[] = {0,1,0,0,0};
    double darkgrayFeature[] = {0,0,1,0,0};
    double redFeature[] = {0,0,0,1,0};
    double yellowFeature[] = {0,0,0,0,1};

    char features[] = {'w','w','w','w','w','w','w',
                       'w','r','r','r','r','g','g',
                       'w','s','w','w','g','w','w',
                       'w','g','g','g','g','y','w',
                       'w','d','w','w','w','y','w',
                       'w','d','w','w','w','y','w',
                       'w','g','g','g','g','g','w',
                       'g','g','w','w','w','w','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(darkgrayFeature, darkgrayFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
            case 'y':
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;

        }
    
    }
    return stateFeatures;


}


//simple world for testing Cakmak algo
double** cakmakBadWorld3(int numFeatures, int width, int height, int numStates)
{
 //hard coded to match paper descriptions
    assert(numFeatures == 7);
    assert(height == 4);
    assert(width == 4);
    assert(numStates == width * height);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double wallFeature[] = {0,0,0,0,0,0,0}; //all zeros for wall since reward doesn't matter
    double starFeature[] = {1,0,0,0,0,0,0};
    double grayFeature[] = {0,1,0,0,0,0,0};
    double darkgrayFeature[] = {0,0,1,0,0,0,0};
    double redFeature[] = {0,0,0,1,0,0,0};
    double yellowFeature[] = {0,0,0,0,1,0,0};
    double blueFeature[] = {0,0,0,0,0,1,0};
    double orangeFeature[] = {0,0,0,0,0,0,1};

    char features[] = {'w','w','r','g',
                       'y','b','s','d',
                       'b','o','y','w',
                       'g','y','b','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(wallFeature, wallFeature+numFeatures, stateFeatures[i]);
                break;
            case 's':
                std::copy(starFeature, starFeature+numFeatures, stateFeatures[i]);
                break;
            case 'd':
                std::copy(darkgrayFeature, darkgrayFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(grayFeature, grayFeature+numFeatures, stateFeatures[i]);
                break;
            case 'y':
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 'b':
                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
                break;
            case 'o':
                std::copy(orangeFeature, orangeFeature+numFeatures, stateFeatures[i]);
                break;

        }
    
    }
    return stateFeatures;


}



void displayStateColorFeatures(double** stateFeatures, unsigned int gridCols, unsigned int gridRows, unsigned int numFeatures)
{
//    double whiteFeature[] = {1,0,0,0,0};
//    double redFeature[] = {0,1,0,0,0};
//    double blueFeature[] = {0,0,1,0,0};
//    double yellowFeature[] = {0,0,0,1,0};
//    double greenFeature[] = {0,0,0,0,1};
//    
    double whiteFeature[] = {1,0,0,0,0,0,0,0};
    double redFeature[] = {0,1,0,0,0,0,0,0};
    double blueFeature[] = {0,0,1,0,0,0,0,0};
    double yellowFeature[] = {0,0,0,1,0,0,0,0};
    double greenFeature[] = {0,0,0,0,1,0,0,0};
    double cyanFeature[] = {0,0,0,0,0,1,0,0};
    double blackFeature[] = {0,0,0,0,0,0,1,0};
    double magentaFeature[] = {0,0,0,0,0,0,0,1};

   unsigned int count = 0;
    for(unsigned int r = 0; r < gridRows; r++)
    {
        for(unsigned int c = 0; c < gridCols; c++)
        {
            if(featuresAreEqual(stateFeatures[count], whiteFeature, numFeatures))
                cout << "w" << "  ";
            else if(featuresAreEqual(stateFeatures[count], redFeature, numFeatures))
                cout << "r" << "  ";
            else if(featuresAreEqual(stateFeatures[count], blueFeature, numFeatures))
                cout << "b" << "  ";
            else if(featuresAreEqual(stateFeatures[count], yellowFeature, numFeatures))
                cout << "y" << "  ";
            else if(featuresAreEqual(stateFeatures[count], greenFeature, numFeatures))
                cout << "g" << "  ";
            else if(featuresAreEqual(stateFeatures[count], cyanFeature, numFeatures))
                cout << "c" << "  ";
            else if(featuresAreEqual(stateFeatures[count], blackFeature, numFeatures))
                cout << "k" << "  ";
            else if(featuresAreEqual(stateFeatures[count], magentaFeature, numFeatures))
                cout << "m" << "  ";

            count++;
        }
        cout << endl;
    }
}

bool featuresAreEqual(double* f1, double* f2, int size)
{
    for(int i=0; i<size; i++)
        if(f1[i] != f2[i]) return false;
    return true;
}

double** randomGridNavDomain(int numStates, int numFeatures)
{
    //hard coded to only allow 5 colors for now
    assert(numFeatures == 5);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0};
    double redFeature[] = {0,1,0,0,0};
    double blueFeature[] = {0,0,1,0,0};
    double yellowFeature[] = {0,0,0,1,0};
    double greenFeature[] = {0,0,0,0,1};
    

                       
    for(int i=0; i < numStates; i++)
    {
        //randomly select one of the colors
        int colorIndex = rand() % numFeatures;
        switch(colorIndex)
        {
            case 0:
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 1:
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 2:
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
            case 3:
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
            case 4:
                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}

//simple world for testing enough is enough concepts
double** enoughToyWorld(int width, int height, int numFeatures, int numStates)
{
 //hard coded to only allow 5 colors for now
    assert(numFeatures == 4);
    assert(width == 3);
    assert(height == 4);
    assert(numStates == width * height);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0};
    double redFeature[] = {0,1,0,0};
    double blueFeature[] = {0,0,1,0};
    double greenFeature[] = {0,0,0,1};

    char features[] = {'w','r','w',
                       'w','b','w',
                       'r','w','r',
                       'g','w','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 'b':
                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
                break;
//            case 'y':
//                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
//                break;
            case 'g':
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;


}

//simple world for testing enough is enough concepts
double** feasibleToyWorld(int width, int height, int numFeatures, int numStates)
{
 //hard coded to only allow 5 colors for now
    assert(numFeatures == 2);
    assert(width == 3);
    assert(height == 4);
    assert(numStates == width * height);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0};
    double redFeature[] = {0,1};

    char features[] = {'w','r','w',
                       'w','w','w',
                       'r','w','r',
                       'w','w','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
        }
    }
    return stateFeatures;


}



//simple world for testing enough is enough concepts
double** improvementToyWorld(int width, int height, int numFeatures, int numStates)
{
 //hard coded to only allow 5 colors for now
    assert(numFeatures == 2);
    assert(width == 3);
    assert(height == 3);
    assert(numStates == width * height);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0};
    double redFeature[] = {0,1};

    char features[] = {'w','w','w',
                       'r','r','w',
                       'w','w','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
        
        }
    
    }
    return stateFeatures;


}

//simple world for testing enough is enough concepts
double** improvementMazeWorld(int width, int height, int numFeatures, int numStates)
{
 //hard coded to only allow 5 colors for now
    assert(numFeatures == 2);
    assert(width == 5);
    assert(height == 5);
    assert(numStates == width * height);
    
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0};
    double redFeature[] = {0,1};

    char features[] = {'w','w','w','r','w',
                       'w','r','w','r','w',
                       'w','r','w','r','w',
                       'w','r','w','r','w',
                       'w','r','w','w','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
        
        }
    
    }
    return stateFeatures;


}



//9x9 world with blue in middle
double** random9x9GridNavGoalWorld()
{
    int numStates = 81;
    int numFeatures = 5;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0};
    double redFeature[] = {0,1,0,0,0};
    double blueFeature[] = {0,0,1,0,0};
    double yellowFeature[] = {0,0,0,1,0};
    double greenFeature[] = {0,0,0,0,1};

                       
    for(int i=0; i < numStates; i++)
    {
        //randomly select one of the colors
        int colorIndex = rand() % (numFeatures-1);
        switch(colorIndex)
        {
            case 0:
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 1:
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 2:
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
            case 3:
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
//            case 4:
//                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
//                break;
        }
    
    }
    //make middle state blue
    std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[40]);
    
    return stateFeatures;


}

//9x9 world with blue in middle
double** cornerGoal6x6GridNavGoalWorld()
{
    int numStates = 36;
    int numFeatures = 5;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0};
    double redFeature[] = {0,1,0,0,0};
    double blueFeature[] = {0,0,1,0,0};
    double yellowFeature[] = {0,0,0,1,0};
    double greenFeature[] = {0,0,0,0,1};

       
    //make everything white at first                
    for(int i=0; i < numStates; i++)
    {
        //randomly select one of the colors
        int colorIndex = rand() % (numFeatures-1);
        std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
       
    }
    std::copy(redFeature, redFeature+numFeatures, stateFeatures[0]);
    std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[5]);
    std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[30]);
    std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[35]);
//                 break;
    //TODO: add four random goal states around the world
//     for(int i=0; i < numFeatures; i++)
//     {
//         //randomly select one of the states to have that color
//         int colorIndex = rand() % (numStates-1);
//         int colorIndex = rand() % (numFeatures-1);
//         switch(colorIndex)
//         {
//             case 0:
//                 std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 1:
//                 std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 2:
//                 std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 3:
//                 std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
//                 break;
// //            case 4:
// //                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
// //                break;
//         }
    
//   }

    
    return stateFeatures;


}

//9x9 world with blue in middle
double** cornerGoal5x5GridNavGoalWorld()
{
    int numStates = 25;
    int numFeatures = 3;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0};
    double redFeature[] = {0,1,0,0,0};
    double blueFeature[] = {0,0,1,0,0};
    //double yellowFeature[] = {0,0,0,1,0};
    //double greenFeature[] = {0,0,0,0,1};

       
    //make everything white at first                
    for(int i=0; i < numStates; i++)
    {
        //randomly select one of the colors
        //int colorIndex = rand() % (numFeatures-1);
        std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
       
    }
    std::copy(redFeature, redFeature+numFeatures, stateFeatures[0]);
    //std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[5]);
    //std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[30]);
    std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[24]);
//                 break;
    //TODO: add four random goal states around the world
//     for(int i=0; i < numFeatures; i++)
//     {
//         //randomly select one of the states to have that color
//         int colorIndex = rand() % (numStates-1);
//         int colorIndex = rand() % (numFeatures-1);
//         switch(colorIndex)
//         {
//             case 0:
//                 std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 1:
//                 std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 2:
//                 std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 3:
//                 std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
//                 break;
// //            case 4:
// //                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
// //                break;
//         }
    
//   }

    
    return stateFeatures;
}

double** cornerGoal5x5_5Features_GridNavGoalWorld()
{
    int numStates = 25;
    int numFeatures = 5;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0};
    double redFeature[] = {0,1,0,0,0};
    double blueFeature[] = {0,0,1,0,0};
    double yellowFeature[] = {0,0,0,1,0};
    double greenFeature[] = {0,0,0,0,1};

       
    //make everything white at first                
    for(int i=0; i < numStates; i++)
    {
        //randomly select one of the colors
        //int colorIndex = rand() % (numFeatures-1);
        std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
       
    }
    std::copy(redFeature, redFeature+numFeatures, stateFeatures[0]);
    std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[4]);
    std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[20]);
    std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[24]);
//                 break;
    //TODO: add four random goal states around the world
//     for(int i=0; i < numFeatures; i++)
//     {
//         //randomly select one of the states to have that color
//         int colorIndex = rand() % (numStates-1);
//         int colorIndex = rand() % (numFeatures-1);
//         switch(colorIndex)
//         {
//             case 0:
//                 std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 1:
//                 std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 2:
//                 std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 3:
//                 std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
//                 break;
// //            case 4:
// //                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
// //                break;
//         }
    
//   }

    
    return stateFeatures;
}

double** cornerGoalNxN_5Features_GridNavGoalWorld(unsigned int size)
{
    int numStates = size*size;
    int numFeatures = 5;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0};
    double redFeature[] = {0,1,0,0,0};
    double blueFeature[] = {0,0,1,0,0};
    double yellowFeature[] = {0,0,0,1,0};
    double greenFeature[] = {0,0,0,0,1};

       
    //make everything white at first                
    for(int i=0; i < numStates; i++)
    {
        //randomly select one of the colors
        //int colorIndex = rand() % (numFeatures-1);
        std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
       
    }
    std::copy(redFeature, redFeature+numFeatures, stateFeatures[0]);
    std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[size-1]);
    std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[size*(size-1)]);
    std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[size*size - 1]);
//                 break;
    //TODO: add four random goal states around the world
//     for(int i=0; i < numFeatures; i++)
//     {
//         //randomly select one of the states to have that color
//         int colorIndex = rand() % (numStates-1);
//         int colorIndex = rand() % (numFeatures-1);
//         switch(colorIndex)
//         {
//             case 0:
//                 std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 1:
//                 std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 2:
//                 std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
//                 break;
//             case 3:
//                 std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
//                 break;
// //            case 4:
// //                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
// //                break;
//         }
    
//   }

    
    return stateFeatures;
}

double** random9x9GridNavGoalWorld8Features()
{
    int numStates = 81;
    int numFeatures = 8;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0,0,0,0};
    double redFeature[] = {0,1,0,0,0,0,0,0};
    double blueFeature[] = {0,0,1,0,0,0,0,0};
    double yellowFeature[] = {0,0,0,1,0,0,0,0};
    double greenFeature[] = {0,0,0,0,1,0,0,0};
    double cyanFeature[] = {0,0,0,0,0,1,0,0};
    double blackFeature[] = {0,0,0,0,0,0,1,0};
    double magentaFeature[] = {0,0,0,0,0,0,0,1};

                       
    for(int i=0; i < numStates; i++)
    {
        //randomly select one of the colors
        int colorIndex = rand() % (numFeatures-1);
        switch(colorIndex)
        {
            case 0:
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 1:
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 2:
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
            case 3:
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
            case 4:
                std::copy(cyanFeature, cyanFeature+numFeatures, stateFeatures[i]);
                break;
            case 5:
                std::copy(blackFeature, blackFeature+numFeatures, stateFeatures[i]);
                break;
            case 6:
                std::copy(magentaFeature, magentaFeature+numFeatures, stateFeatures[i]);
                break;
//            case 4:
//                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
//                break;
        }
    
    }
    //make middle state blue
    std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[40]);
    
    return stateFeatures;


}

double** random7x7GridNavGoalWorld4Features()
{
    int numStates = 49;
    int numFeatures = 4;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0};
    double redFeature[] = {0,1,0,0};
    double blueFeature[] = {0,0,1,0};
    double greenFeature[] = {0,0,0,1};

                       
    for(int i=0; i < numStates; i++)
    {
        //randomly select one of the colors
        int colorIndex = rand() % (numFeatures-1);
        switch(colorIndex)
        {
            case 0:
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 1:
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 2:
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    //make middle state blue
    std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[24]);
    
    return stateFeatures;


}




//No terminal state
double** random9x9GridNavWorld8Features()
{
    int numStates = 81;
    int numFeatures = 8;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0,0,0,0};
    double redFeature[] = {0,1,0,0,0,0,0,0};
    double blueFeature[] = {0,0,1,0,0,0,0,0};
    double yellowFeature[] = {0,0,0,1,0,0,0,0};
    double greenFeature[] = {0,0,0,0,1,0,0,0};
    double cyanFeature[] = {0,0,0,0,0,1,0,0};
    double blackFeature[] = {0,0,0,0,0,0,1,0};
    double magentaFeature[] = {0,0,0,0,0,0,0,1};

                       
    for(int i=0; i < numStates; i++)
    {
        //randomly select one of the colors
        int colorIndex = rand() % numFeatures;
        switch(colorIndex)
        {
            case 0:
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 1:
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 2:
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
            case 3:
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
            case 4:
                std::copy(cyanFeature, cyanFeature+numFeatures, stateFeatures[i]);
                break;
            case 5:
                std::copy(blackFeature, blackFeature+numFeatures, stateFeatures[i]);
                break;
            case 6:
                std::copy(magentaFeature, magentaFeature+numFeatures, stateFeatures[i]);
                break;
            case 7:
                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
   
    return stateFeatures;


}



//No terminal state
double** randomNxNGridNavWorldXFeatures(int N, int X)
{
    int numStates = N*N;
    int numFeatures = X;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
    {
        stateFeatures[i] = new double[numFeatures];
        for(int f = 0; f < numFeatures; f++)
            stateFeatures[i][f] = 0.0;
    }
        
                         
    for(int i=0; i < numStates; i++)
    {
        
        //randomly select one of the colors
        int f = rand() % numFeatures;
        stateFeatures[i][f] = 1.0;
    
    }
   
    return stateFeatures;


}

//No terminal state 
double** randomNxNGridNavWorldMixedFeatures(int N, int F)
{
    int numStates = N*N;
    int numFeatures = F;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
    {
        stateFeatures[i] = new double[numFeatures];
        //draw random feature values for each state
        for(int f = 0; f < numFeatures; f++)
            stateFeatures[i][f] = ((double) rand() / (RAND_MAX));
    }
        
    return stateFeatures;


}




//No terminal state
double** randomNxNGridNavWorld8Features(int N)
{
    int numStates = N*N;
    int numFeatures = 8;
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0,0,0,0};
    double redFeature[] = {0,1,0,0,0,0,0,0};
    double blueFeature[] = {0,0,1,0,0,0,0,0};
    double yellowFeature[] = {0,0,0,1,0,0,0,0};
    double greenFeature[] = {0,0,0,0,1,0,0,0};
    double cyanFeature[] = {0,0,0,0,0,1,0,0};
    double blackFeature[] = {0,0,0,0,0,0,1,0};
    double magentaFeature[] = {0,0,0,0,0,0,0,1};

                       
    for(int i=0; i < numStates; i++)
    {
        //randomly select one of the colors
        int colorIndex = rand() % numFeatures;
        switch(colorIndex)
        {
            case 0:
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 1:
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 2:
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
            case 3:
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
            case 4:
                std::copy(cyanFeature, cyanFeature+numFeatures, stateFeatures[i]);
                break;
            case 5:
                std::copy(blackFeature, blackFeature+numFeatures, stateFeatures[i]);
                break;
            case 6:
                std::copy(magentaFeature, magentaFeature+numFeatures, stateFeatures[i]);
                break;
            case 7:
                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
   
    return stateFeatures;


}


//TODO make sure to delete this!!!
//Set's up stateFeatures for simple grid world that I've been playing with over the 
//last couple of weeks
double** initFeaturesToyFeatureDomain5x5(int numStates, int numFeatures)
{
     if(numStates != 25 || numFeatures != 5) 
     {
        cout << "[ERROR] This domain only works for 5x5 with 5 features!" << endl;
        return nullptr;
     }
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0};
    double redFeature[] = {0,1,0,0,0};
    double blueFeature[] = {0,0,1,0,0};
    double yellowFeature[] = {0,0,0,1,0};
    double greenFeature[] = {0,0,0,0,1};
    
    char features[] = {'w','g','y','r','w',
                       'w','r','w','w','w',
                       'w','w','b','w','g',
                       'y','g','w','w','w',
                       'w','w','r','y','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 'b':
                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
                break;
            case 'y':
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;
}

//example of how worst-case and simple feature count differences don't work
double** initFeatureCountToyDomain4x4(int numStates, int numFeatures)
{
     if(numStates != 16 || numFeatures != 4) 
     {
        cout << "[ERROR] This domain only works for 4x4 with 4 features!" << endl;
        return nullptr;
     }
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0};
    double blueFeature[] = {0,1,0,0};
    double greenFeature[] = {0,0,1,0};
    double yellowFeature[] = {0,0,0,1};
    
    char features[] = {'w','w','w','w',
                       'w','g','g','w',
                       'b','g','g','w',
                       'b','b','w','y'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 'b':
                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
                break;
            case 'y':
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;
}


//example of how worst-case and simple feature count differences don't work
double** initFeatureCountDebugDomain2x2(int numStates, int numFeatures)
{
     if(numStates != 4 || numFeatures != 3) 
     {
        cout << "[ERROR] This domain only works for 2x2 with 3 features!" << endl;
        return nullptr;
     }
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0};
    double greenFeature[] = {0,1,0};
    double blueFeature[] = {0,0,1};
    
    char features[] = {'w','w',
                       'g','b'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 'b':
                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;
}

double** initRandomFeaturesRandomDomain(const int numStates, const int numFeatures)
{
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
    {
        stateFeatures[i] = new double[numFeatures];
        double feature[numFeatures];
        for (int feat=0; feat < numFeatures; feat++) feature[feat] = (double)(rand() % 2);
        std::copy(feature, feature+numFeatures, stateFeatures[i]);
    }
    return stateFeatures;
}

//example of how worst-case and simple feature count differences don't work
double** sparse_barrier4x4(int numStates, int numFeatures)
{
     if(numStates != 16 || numFeatures != 2) 
     {
        cout << "[ERROR] This domain only works for 4x4 with 2 features!" << endl;
        return nullptr;
     }
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0};
    double redFeature[] = {0,1};
    
    
    char features[] = {'w','w','w','w',
                       'w','w','w','w',
                       'w','w','r','w',
                       'w','w','w','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;
}

//example of how worst-case and simple feature count differences don't work
double** sparse_barrier9x9(int numStates, int numFeatures)
{
     if(numStates != 81 || numFeatures != 5) 
     {
        cout << "[ERROR] This domain only works for 9x9 with 5 features!" << endl;
        return nullptr;
     }
    double** stateFeatures;
    stateFeatures = new double*[numStates];
    for(int i=0; i<numStates; i++)
        stateFeatures[i] = new double[numFeatures];
        
    double whiteFeature[] = {1,0,0,0,0};
    double redFeature[] = {0,1,0,0,0};
    double blueFeature[] = {0,0,1,0,0};
    double yellowFeature[] = {0,0,0,1,0};
    double greenFeature[] = {0,0,0,0,1};
    
    
    char features[] = {'w','w','w','w','w','w','w','w','w',
                       'w','w','r','w','w','w','b','w','w',
                       'w','r','r','w','w','w','b','b','w',
                       'w','w','w','w','w','w','w','w','w',
                       'w','w','w','w','w','w','w','w','w',
                       'w','w','w','w','w','w','w','w','w',
                       'w','y','y','w','w','w','g','g','w',
                       'w','w','y','w','w','w','g','w','w',
                       'w','w','w','w','w','w','w','w','w'};
                       
    for(int i=0; i < numStates; i++)
    {
        switch(features[i])
        {
            case 'w':
                std::copy(whiteFeature, whiteFeature+numFeatures, stateFeatures[i]);
                break;
            case 'r':
                std::copy(redFeature, redFeature+numFeatures, stateFeatures[i]);
                break;
            case 'b':
                std::copy(blueFeature, blueFeature+numFeatures, stateFeatures[i]);
                break;
            case 'y':
                std::copy(yellowFeature, yellowFeature+numFeatures, stateFeatures[i]);
                break;
            case 'g':
                std::copy(greenFeature, greenFeature+numFeatures, stateFeatures[i]);
                break;
        }
    
    }
    return stateFeatures;
}




#endif
