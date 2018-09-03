
#ifndef driving_world_h
#define driving_world_h

#include "SDL/SDL.h"
#include "SDL/SDL_image.h"
#include "SDL/SDL_ttf.h"
#include <string>
#include <map>
#include <iostream>
#include <cmath>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <iomanip> // setprecision
#include <sstream> // stringstream
#include <vector>

using namespace std;


enum ACTION { LEFT, STAY, RIGHT };

//world attributes
        //TODO make this configurable
        //const int NUM_OTHER_CARS = 1;
        
        const int DEMO_LENGTH = 100;
        
        const int NUM_LANES = 5;
        const int NUM_ACTIONS = 3;

        const int SCREEN_WIDTH = 800;
        const int SCREEN_HEIGHT = 500;  //6 car widths
        //const int SCREEN_HEIGHT = 300;  //6 car widths
        const int SCREEN_BPP = 32;
        const int LANE_WIDTH = SCREEN_WIDTH / 8;
        //The attributes of the car
        const int CAR_WIDTH = 30;
        const int CAR_HEIGHT = 60;
        const double CAR_SPEED = 1.0; // 1 car heights per step



        //TODO make it so any number of cars works
        //bool two_cars = true; //use two bad cars 
        
        ////Stuff for world        
        //The wall
        SDL_Rect left_wall;
        SDL_Rect right_wall;
        SDL_Rect road;
        SDL_Rect offroad_left;
        SDL_Rect offroad_right;

        //the lines
        //lines for lanes
        SDL_Rect line1, line2;
        //lines for offroad
        SDL_Rect offroad_line1, offroad_line2;

        ////The cars' collision boxes
        SDL_Rect box;             //me
        SDL_Rect bad_car_box;
        SDL_Rect bad_car_box2;

        //The velocity of the car
        int xVel, yVel;

            
        
        ////The surfaces and stuff for viz
        SDL_Surface *car = NULL;
        SDL_Surface *bad_car = NULL;
        SDL_Surface *bad_car2 = NULL;
        SDL_Surface *message = NULL;

        SDL_Surface *screen = NULL;

        //The font
        TTF_Font *font = NULL;

        //The color of the font
        SDL_Color textColor = { 0, 0, 0 };

        //The event structure
        SDL_Event event;






SDL_Surface *load_image( std::string filename )
{
  //The image that's loaded
  SDL_Surface* loadedImage = NULL;

  //The optimized surface that will be used
  SDL_Surface* optimizedImage = NULL;

  //Load the image
  loadedImage = IMG_Load( filename.c_str() );

  //If the image loaded
  if( loadedImage != NULL )
  {
    //Create an optimized surface
    optimizedImage = SDL_DisplayFormat( loadedImage );

    //Free the old surface
    SDL_FreeSurface( loadedImage );

    //If the surface was optimized
    if( optimizedImage != NULL )
    {
      //Color key surface
      SDL_SetColorKey( optimizedImage, SDL_SRCCOLORKEY, SDL_MapRGB( optimizedImage->format, 0, 0xFF, 0xFF ) );
    }
  }

  //Return the optimized surface
  return optimizedImage;
}




//TODO:
//hold lane numbers for all cars, including me
//hold distances from my car to all cars
class State
{
    private:
        double distToCarInMyLane;
        //TODO
        vector<int> stateFeatures; //features for Q-learning
        vector<int> rewardFeatures; //features for Phi(s)
    public:
        //constructor
        State(double* carDists, int myLane, int numStateFeatures, int numRewardFeatures)
        {
            assert(numStateFeatures == 12);
            assert(numRewardFeatures == 6);
//            //***debugging
//            cout << "Lane dists: ";
//            for(int i=0; i<NUM_LANES;i++)
//                cout << carDists[i] << ",";
//            cout << endl;
//            cout << "my lane: " << myLane << endl;
//            //***
        
            //process state into stateFeatures
            stateFeatures.resize(numStateFeatures);
            //initialize to all zero 
            for(unsigned int i=0;i<stateFeatures.size();i++) stateFeatures[i] = 0;
            //Turn on active features
            //Feature 0 (in Collision)
            if(carDists[myLane] >= -2 * CAR_HEIGHT)
            {
                if(carDists[myLane] <= 0) 
                    stateFeatures[0] = 1;
                //Feature 1 (going to hit next time step)
                else if(carDists[myLane] <= CAR_HEIGHT * CAR_SPEED)
                    stateFeatures[1] = 1;
                //Feature 2 (going to hit in two time steps)
                else if(carDists[myLane] <= 2 * CAR_HEIGHT * CAR_SPEED)
                    stateFeatures[2] = 1;
            }
//            Feature 3-7 (which lane the car is in)
            stateFeatures[myLane + 3] = 1;
//                features[myLane] = 1;
//            //Feature 8 (if I turn left will I collide?)
            if(myLane > 0)  //Is there a left lane and will I hit it next step?
                if(carDists[myLane - 1] <= CAR_HEIGHT * CAR_SPEED) 
                    if(carDists[myLane - 1] >= -1 * CAR_HEIGHT * CAR_SPEED)
                        stateFeatures[8] = 1;
            //Feature 9 (if I turn left will I tailgate?, i.e., be one step away from hitting?)
            if(myLane > 0)  
                if(carDists[myLane - 1] <= 2 * CAR_HEIGHT * CAR_SPEED)
                    if(carDists[myLane - 1] > CAR_HEIGHT * CAR_SPEED)
                        stateFeatures[9] = 1;
            //Feature 10 (if I turn right will I collide?)
            if(myLane < 4)  //Is there a right lane and will I hit it next step?
                if(carDists[myLane + 1] <= CAR_HEIGHT * CAR_SPEED) 
                    if(carDists[myLane + 1] >= -1 * CAR_HEIGHT * CAR_SPEED)
                        stateFeatures[10] = 1;
            if(myLane < 4)  
                if(carDists[myLane + 1] <= 2 * CAR_HEIGHT * CAR_SPEED)
                    if(carDists[myLane + 1] > CAR_HEIGHT * CAR_SPEED)
                        stateFeatures[11] = 1;
            
                    
            //process state into rewardFeatures
            rewardFeatures.resize(numRewardFeatures);
            //initialize to all zero 
            for(unsigned int i=0;i<rewardFeatures.size();i++) rewardFeatures[i] = 0;
            //Turn on active features
            //Feature 0 (in Collision)
            if(carDists[myLane] <= 0 && carDists[myLane] >= -2 * CAR_HEIGHT) 
                rewardFeatures[0] = 1;
            //Feature 1 (going to hit next time step)
            //else if(carDists[myLane] <= CAR_HEIGHT * CAR_SPEED && carDists[myLane] > 0)
            //    rewardFeatures[1] = 1;
            //Feature 2-6 (which lane the car is in)
            rewardFeatures[myLane + 1] = 1;
            
        };
        //destructor
        ~State()
        {

        }
        //getter
        double getDistToCarInMyLane() { return distToCarInMyLane; };
        //setter        
        void setDistToCarInMyLane(double newDist) { distToCarInMyLane = newDist; }; 
 
        unsigned int getNumStateFeatures()
        {
            return stateFeatures.size();
        }
        unsigned int getNumRewardFeatures()
        {
            return rewardFeatures.size();
        }
        
        vector<int> getStateFeatures()
        {
            return stateFeatures;
        };
        
        double getStateFeature(int i)
        {
            return stateFeatures[i];
        };
        
        vector<int> getRewardFeatures()
        {
            return rewardFeatures;
        }
        double getRewardFeature(int i)
        {
            return rewardFeatures[i];
        }
 
        //toString method for debugging       
        string toStateString(){
//            string stateInfo = "[";
//            for(unsigned int i=0; i<stateFeatures.size()-1; i++)
//                stateInfo += to_string(stateFeatures[i]) + ", ";
//            stateInfo += to_string(stateFeatures[stateFeatures.size()-1]) + "]";
//            return stateInfo;
            char state[stateFeatures.size() + 1];
            for(unsigned int i=0; i<stateFeatures.size(); i++)
                state[i] = stateFeatures[i] +'0';
            state[stateFeatures.size()] = '\0';
            return string(state);
        }
        
        string toRewardString(){
//            string stateInfo = "[";
//            for(unsigned int i=0; i<rewardFeatures.size()-1; i++)
//                stateInfo += to_string(rewardFeatures[i]) + ", ";
//            stateInfo += to_string(rewardFeatures[rewardFeatures.size()-1]) + "]";
//            return stateInfo;
            char reward[rewardFeatures.size() + 1];
            for(unsigned int i=0; i<rewardFeatures.size(); i++)
                reward[i] = rewardFeatures[i] +'0';
            reward[rewardFeatures.size()] = '\0';
            return string(reward);
        
        
        }
        
};


class DrivingWorld 
{
    private:
        bool visualize;  //set to true to see the simulation
        void displayWorld();

        
        void updateFrame();
        void initializeRoadPositions();
        
        bool load_files();
        bool init();
        
       double getCarDistance(SDL_Rect A, SDL_Rect B);
       double getDistanceToClosestCarInLane(int laneNumber);
       double* featureWeights;
       int numStateFeatures;
       int numRewardFeatures;

    public:
   
           //reward?
        float score = 0;
        bool two_cars;
        bool human_demo;

        //constructor
        DrivingWorld(bool vizOn, double* fWeights, int nStateFeatures, int nRewardFeatures, bool twoCars=false, bool humanDemo=false): visualize(vizOn), numStateFeatures(nStateFeatures), numRewardFeatures(nRewardFeatures), two_cars(twoCars), human_demo(humanDemo)
        {
            //copy feature weights
            featureWeights = new double[nRewardFeatures];
            for(int i=0; i<nRewardFeatures; i++)
                featureWeights[i] = fWeights[i];
            //initialize stuff for world
            //initialize current state
             
        };
        //TODO destructor
        ~DrivingWorld()
        {
            delete[] featureWeights;    
        };
        
        void setFeatureWeights(double* newWeights)
        {
            for(int i=0; i<numRewardFeatures; i++)
                featureWeights[i] = newWeights[i];
        };

        double* getFeatureWeights()
        {
            return featureWeights;
        };
        
        void printFeatureWeights()
        {
            for(int i=0;i<numRewardFeatures; i++)
                cout << featureWeights[i] << "\t";
            cout << endl;
        };
        
        //return <next-state, reward> pair for taking action in current state
        pair<State,double> updateState(unsigned int action);
        
        int getNumRewardFeatures(){ return numRewardFeatures; };
        
        int getNumStateFeatures(){ return numStateFeatures; };
        //resets the game and returns the starting state
        State startNewEpoch();
        
        //TODO return current reward for being in state as linear combination of features
        double getReward(State s);
        
        void setVisuals(bool vizOn) { visualize = vizOn; };
        void closeScreen() {  };
        
        //TODO return current features
        double getCurrentFeatures();
         void clean_up();  
        //TODO return actual state features
        //The collision box of the car
        SDL_Rect box;
        SDL_Rect bad_car_box;
        SDL_Rect bad_car_box2;

        //The velocity of the car
        int xVel, yVel;

        //Initializes the variables
        void initializeCarPositions();
        


        //Takes key presses and adjusts the car's velocity
        int handle_input();

        //Moves the car
        void move();

        //Shows the car on the screen
        void show();

        int getCurrentLaneNumber(int car_index);
        int getCurrentDistanceFromNearestCar();
        bool leftLaneHasCar();
        bool rightLaneHasCar();
        State getCurrentState();
        bool isCollision();
        bool check_collision( SDL_Rect A, SDL_Rect B );
        void apply_surface( int x, int y, SDL_Surface* source, SDL_Surface* destination, SDL_Rect* clip = NULL );
        int getNumActions(){ return NUM_ACTIONS; };
        vector<pair<string,unsigned int> > startHumanDemo();
            

};

double DrivingWorld::getReward(State s)
{
    //perform dot product with state features and featureWeights
    double reward = 0;
    for(int i=0; i<numRewardFeatures; i++)
        reward += featureWeights[i] * s.getRewardFeature(i);
    return reward;
}


void DrivingWorld::initializeCarPositions()
{
  //Initialize the offsets
  //agent car starts in lane 2
  box.x = 2*LANE_WIDTH + LANE_WIDTH/2 - CAR_WIDTH/2;
  //box.y = 4 * CAR_HEIGHT;
  box.y = 6 * CAR_HEIGHT;

  //start in lane 1
  bad_car_box.x = 1*LANE_WIDTH + LANE_WIDTH/2 - CAR_WIDTH/2;
  bad_car_box.y = 4*CAR_HEIGHT;
  ////start in lane 2
  //bad_car_box2.x = 2*LANE_WIDTH + LANE_WIDTH/2 - CAR_WIDTH/2;
  //start in lane 3
  bad_car_box2.x = 3*LANE_WIDTH + LANE_WIDTH/2 - CAR_WIDTH/2;
  bad_car_box2.y = 0;

  //Set the car's dimentions
  box.w = CAR_WIDTH;
  box.h = CAR_HEIGHT;
  bad_car_box.w = CAR_WIDTH;
  bad_car_box.h = CAR_HEIGHT;
  bad_car_box2.w = CAR_WIDTH;
  bad_car_box2.h = CAR_HEIGHT;

  //Initialize the velocity
  xVel = 0;
  yVel = -CAR_HEIGHT * CAR_SPEED;
}

int DrivingWorld::getCurrentLaneNumber(int car)
{
  int xPos = 0;
  //cout << "box.x: " << box.x <<endl;
  if (car == 0)
  {
    xPos = box.x;
  }
  else if (car == 1)
  {
    xPos = bad_car_box.x;
  }
  else if (car == 2)
  {
    xPos = bad_car_box2.x;
  }
    
  if(xPos < 1*LANE_WIDTH ) return 0;
  if(xPos < 2*LANE_WIDTH ) return 1;
  if(xPos < 3*LANE_WIDTH ) return 2;
  if(xPos < 4*LANE_WIDTH ) return 3;
  else return 4;


}


int DrivingWorld::getCurrentDistanceFromNearestCar()
{
  if(getCurrentLaneNumber(0) == getCurrentLaneNumber(1))
  {
    int dist = (int) ((box.y - bad_car_box.y)/CAR_HEIGHT - 1);
    if (dist >= 0 and dist <= 2) return dist;
    else return 2;
  }
  else if(two_cars && getCurrentLaneNumber(0) == getCurrentLaneNumber(2))
  {
    return (int) ((box.y - bad_car_box2.y)/CAR_HEIGHT);
  }
  else{
    return 2;//(SCREEN_HEIGHT/CAR_HEIGHT - 1);
  }

}
//TODO check logic
bool DrivingWorld::leftLaneHasCar()
{
  if(getCurrentLaneNumber(0) == 0) return true;
  if(bad_car_box.y < 3 * SCREEN_HEIGHT / 4 - CAR_HEIGHT && (!two_cars || bad_car_box2.y < 3 * SCREEN_HEIGHT / 4 - CAR_HEIGHT)) return false;
  if(getCurrentLaneNumber(0) == getCurrentLaneNumber(1) + 1 || (two_cars && (getCurrentLaneNumber(0) == getCurrentLaneNumber(2) + 1))) return true;

  return false;
}
//TODO check logic
bool DrivingWorld::rightLaneHasCar()
{
  if(getCurrentLaneNumber(0) == 2) return true;
  if(bad_car_box.y < 3 * SCREEN_HEIGHT / 4 - CAR_HEIGHT/2 && (!two_cars ||bad_car_box2.y < 3 * SCREEN_HEIGHT / 4 - CAR_HEIGHT/2)) return false;
  if(getCurrentLaneNumber(0) == getCurrentLaneNumber(1) - 1 || (two_cars && (getCurrentLaneNumber(0) == getCurrentLaneNumber(2) - 1))) return true;

  return false;
}

//get distance from top of A to bottom of B
double DrivingWorld::getCarDistance(SDL_Rect A, SDL_Rect B)
{
  double topA = A.y;
  double bottomB = B.y + B.h;
  return topA - bottomB;

}

//TODO make sure assumptions hold by adding asserts for initial car placements, etc.
//pick closest, where within CAR_HEIGHT in negative direction is collision
    //assumes cars are spaced at least two car heights apart so if hitting someone that is the closest car and you have time to avoid the next car in lane after crashing
    
double DrivingWorld::getDistanceToClosestCarInLane(int laneNumber)
{
    int bad_car_lane = getCurrentLaneNumber(1);
    int bad_car_lane2 = getCurrentLaneNumber(2); 
    double bad_car_dist = SCREEN_HEIGHT;
    double bad_car2_dist = SCREEN_HEIGHT;
    //assume if within car_height, then it is closest
    if(laneNumber == bad_car_lane)
    {
        bad_car_dist =  getCarDistance(box, bad_car_box);
        if(abs(bad_car_dist) <= 2 * CAR_HEIGHT)
            return bad_car_dist;
    }
    if(laneNumber == bad_car_lane2)
    {
        bad_car2_dist = getCarDistance(box, bad_car_box2);
        if(abs(bad_car2_dist) <= 2 * CAR_HEIGHT)
            return bad_car2_dist;

    }
    vector<double> dists;
    dists.push_back(bad_car_dist);
    dists.push_back(bad_car2_dist);
    //otherwise find one with smallest positive distance (distance in front of me)
    double minDist = SCREEN_HEIGHT;
    for(double d : dists)
        if(d >=0 && d < minDist)
            minDist = d;     
    
    return minDist;

}

State DrivingWorld::getCurrentState()
{
  int myLane = getCurrentLaneNumber(0);
  double laneDists[NUM_LANES];
  for(int lane = 0; lane < NUM_LANES; lane++)
    laneDists[lane] = getDistanceToClosestCarInLane(lane);
  return State(laneDists, myLane, numStateFeatures, numRewardFeatures);
   
}

bool DrivingWorld::isCollision()
{
    if(check_collision(box, bad_car_box ) || check_collision(box, bad_car_box2 ))
        return true;
    return false;
}

void DrivingWorld::move()
{
  

  //Move the car left or right
  box.x += xVel * LANE_WIDTH;
  if( box.x > right_wall.x) 
  {
 //   cout << "collision with right wall" << endl;
    box.x = right_wall.x - LANE_WIDTH/2 - CAR_WIDTH/2;
  }
  if( box.x < left_wall.x) 
  {
 //   cout << "collision with left wall" << endl;
    box.x = left_wall.x + LANE_WIDTH/2 - CAR_WIDTH/2;
  }
  bad_car_box.y -=  yVel;
  if(two_cars)
      bad_car_box2.y -=  yVel;


  //if(bad_car_box.y < 0) bad_car_box.y = SCREEN_HEIGHT;

  //respawn if in same lane as other car
  while((bad_car_box.y > SCREEN_HEIGHT + CAR_HEIGHT)  || (two_cars && (getCurrentLaneNumber(1) == getCurrentLaneNumber(2)))){
    bad_car_box.y = 0;
    //pick lane 1,2, or 3 to respawn
    bad_car_box.x = (1 + rand() % 3) * LANE_WIDTH + LANE_WIDTH/2 - CAR_WIDTH/2;
  

  }
   

  //if( bad_car_box2.y < 0) bad_car_box2.y = SCREEN_HEIGHT;
  if(two_cars)
  {
      while( bad_car_box2.y > SCREEN_HEIGHT + CAR_HEIGHT || (getCurrentLaneNumber(1) == getCurrentLaneNumber(2))){
        bad_car_box2.y = 0;
        bad_car_box2.x = (1 + rand() % 3) * LANE_WIDTH + LANE_WIDTH/2 - CAR_WIDTH/2;
       
      }
     
  }

}

void DrivingWorld::show()
{
  //Show the car
  DrivingWorld::apply_surface( box.x, box.y, car, screen );
  apply_surface( bad_car_box.x, bad_car_box.y, bad_car, screen );
  if(two_cars) apply_surface( bad_car_box2.x, bad_car_box2.y, bad_car2, screen );
}



//reset game and return starting state
vector<pair<string,unsigned int> > DrivingWorld::startHumanDemo()
{

    //TODO store human demo in this vector
    vector<pair<string,unsigned int> > demonstration;

    //reset score
    score = 0;
    
    
      //Quit flag
  bool quit = false;

    //initialize viz and state code
//always visualize human demo    
//    if(visualize)
//    {
      //Initialize
      if( init() == false )
      {
        cout << "can't init" << endl;
      }

      //Load the files
      if( load_files() == false )
      {
        cout << "can't load files" << endl;
      }

      //Set the message
      message = TTF_RenderText_Solid( font, "Score: 0", textColor );
   // }
    initializeCarPositions();
    initializeRoadPositions();
    
  //  int lane = getCurrentLaneNumber(0);
  //  cout << "Starting lane: " << lane << endl;

  
  int steps = 0;
  //While the user hasn't quit
  while( quit == false && steps < DEMO_LENGTH)
  {
    
    cout << "STEPS: "<<  steps << endl;
    steps++;
    State curState = getCurrentState();
    int action = 1;
  
    //While there's events to handle
    while( SDL_PollEvent( &event ) )
    {
      //Handle events for the car
      action = handle_input();

      //If the user has Xed out the window
      SDL_PollEvent( &event );
      if( event.type == SDL_QUIT )
      {
        //Quit the program
        quit = true;
      }
    }

    //update demo with state-action pair    
    demonstration.push_back(make_pair(curState.toStateString(), action));

    //make a move to update simulator
    
    switch(action)
    {
        case LEFT: 
            xVel = -1.0;
            break;
        case STAY:
            xVel = 0.0;
            break;
        case RIGHT:
            xVel = +1.0;
            break;
    }

    //Move the car
    move();
    
    

    //if(visualize)
    //{
        updateFrame();   
        //Update the screen
        if( SDL_Flip( screen ) == -1 )
        {
          cout << "can't update screen" << endl;
        }   
        usleep(100000);
    //}
    }
    return demonstration;
}


//reset game and return starting state
State DrivingWorld::startNewEpoch()
{
    //reset score
    score = 0;

    //initialize viz and state code

    if(visualize)
    {
      //Initialize
      if( init() == false )
      {
        cout << "can't init" << endl;
      }

      //Load the files
      if( load_files() == false )
      {
        cout << "can't load files" << endl;
      }

      //Set the message
      message = TTF_RenderText_Solid( font, "Score: 0", textColor );
    }
    initializeCarPositions();
    initializeRoadPositions();
    
  //  int lane = getCurrentLaneNumber(0);
  //  cout << "Starting lane: " << lane << endl;


    return getCurrentState();
}

//take action and update state and return reward
//TODO return reward based on features of current state and feature weights
pair<State, double> DrivingWorld::updateState(unsigned int action)
{



    
    //make a move
    switch(action)
    {
        case LEFT: 
            xVel = -1.0;
   //         cout << "Taking action LEFT" << endl;
            break;
        case STAY:
            xVel = 0.0;
  //          cout << "Taking action STAY" << endl;
            break;
        case RIGHT:
            xVel = +1.0;
  //          cout << "Taking action RIGHT" << endl;
            break;
    }

    //Move the car
    move();
    
    
    //create State object with updated state info
    //TODO
    State nextState = getCurrentState();
    
    double reward = getReward(nextState); 
    score += reward;   
//    //check collisions and update score
//    if(isCollision())
//    {
//        cout << "collision with car" << endl;
//        reward -= 1;
//        score += reward;

//    }
//    //cout << score << endl;
//    
    
    
//  cout << "Current State: (" << nextState.toString() << ")" << endl;
//    int lane = getCurrentLaneNumber(0);
//  cout << "Next lane: " << lane << endl;

    if(visualize)
    {
        updateFrame();   
        //Update the screen
        if( SDL_Flip( screen ) == -1 )
        {
          cout << "can't update screen" << endl;
        }   
        usleep(100000);
    }
    //add a delay to see what is happening (delay is in millionths of seconds)
    
    
 //   }


    return make_pair(nextState, reward);       

}


int DrivingWorld::handle_input()
{
  int action = 1;
  //If a key was released
  if( event.type == SDL_KEYDOWN )
  {
    //Adjust the velocity
    switch( event.key.keysym.sym )
    {
      case SDLK_LEFT: action = 0; break;
      case SDLK_RIGHT: action = 2; break;
    }
  }
  return action;
}

void DrivingWorld::updateFrame()
{
        //cout << "updating frame" << endl;
 //Fill the screen white
        SDL_FillRect( screen, &screen->clip_rect, SDL_MapRGB( screen->format, 0xFF, 0xFF, 0xFF ) );
        SDL_FillRect( screen, &road, SDL_MapRGB( screen->format, 0x00, 0x00, 0x00 ) );
        SDL_FillRect( screen, &offroad_left, SDL_MapRGB( screen->format, 0x00, 0x44, 0x00 ) );
        SDL_FillRect( screen, &offroad_right, SDL_MapRGB( screen->format, 0x00, 0x44, 0x00 ) );

        SDL_FillRect( screen, &line1, SDL_MapRGB( screen->format, 0xFF, 0xFF, 0xFF ) );
        SDL_FillRect( screen, &line2, SDL_MapRGB( screen->format, 0xFF, 0xFF, 0xFF ) );
        SDL_FillRect( screen, &offroad_line1, SDL_MapRGB( screen->format, 0xFF, 0xFF, 0x00 ) );
        SDL_FillRect( screen, &offroad_line2, SDL_MapRGB( screen->format, 0xFF, 0xFF, 0x00 ) );

        //Show the wall
        SDL_FillRect( screen, &left_wall, SDL_MapRGB( screen->format, 0x77, 0x77, 0x77 ) );
        SDL_FillRect( screen, &right_wall, SDL_MapRGB( screen->format, 0x77, 0x77, 0x77 ) );

        //Show the car on the screen
        show();

        //update the message
        string score_msg = "Score: ";
        stringstream score_val;
        score_val << fixed << setprecision(2) << score;
        score_msg += score_val.str();
        message = TTF_RenderText_Solid( font, score_msg.c_str(), textColor );

        //Show the message
        apply_surface( LANE_WIDTH * 5 + 30, message->h, message, screen );

        



}

void DrivingWorld::apply_surface( int x, int y, SDL_Surface* source, SDL_Surface* destination, SDL_Rect* clip)
{
  //Holds offsets
  SDL_Rect offset;

  //Get offsets
  offset.x = x;
  offset.y = y;

  //Blit
  SDL_BlitSurface( source, clip, destination, &offset );
}

bool DrivingWorld::check_collision( SDL_Rect A, SDL_Rect B )
{
  //The sides of the rectangles
  int leftA, leftB;
  int rightA, rightB;
  int topA, topB;
  int bottomA, bottomB;

  //Calculate the sides of rect A
  leftA = A.x;
  rightA = A.x + A.w;
  topA = A.y;
  bottomA = A.y + A.h;

  //Calculate the sides of rect B
  leftB = B.x;
  rightB = B.x + B.w;
  topB = B.y;
  bottomB = B.y + B.h;

  //If any of the sides from A are outside of B
  if( bottomA < topB )
  {
    return false;
  }

  if( topA > bottomB )
  {
    return false;
  }

  if( rightA < leftB )
  {
    return false;
  }

  if( leftA > rightB )
  {
    return false;
  }
  //If none of the sides from A are outside B
  return true;
}

bool DrivingWorld::init()
{
  //Initialize all SDL subsystems
  if( SDL_Init( SDL_INIT_EVERYTHING ) == -1 )
  {
    return false;
  }

  //Set up the screen
  screen = SDL_SetVideoMode( SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_BPP, SDL_SWSURFACE );

  //If there was an error in setting up the screen
  if( screen == NULL )
  {
    return false;
  }

   //Initialize SDL_ttf
    if( TTF_Init() == -1 )
    {
        return false;
    }

  //Set the window caption
  SDL_WM_SetCaption( "Driving Task Simulation", NULL );

  //If everything initialized fine
  return true;
}

bool DrivingWorld::load_files()
{
  //Load the car image
  car = load_image( "image/tinycar.png" );
  bad_car = load_image("image/tinyredcar.png");
  bad_car2 = load_image("image/tinyredcar.png");

  //If there was a problem in loading the car
  if( car == NULL || bad_car == NULL)
  { 
    cout << "Couldn't load car images" << endl;
    return false;
  }

  //Open the font
  font = TTF_OpenFont( "Times_New_Roman_Bold.ttf", 40 );

  //If there was an error in loading the font
  if( font == NULL )
  {
    cout << "couldn't load font" << endl;
    return false;
  }

  //If everything loaded fine
  return true;
}

void DrivingWorld::clean_up()
{
  //Free the surface
  SDL_FreeSurface( car );
  SDL_FreeSurface( bad_car );
  SDL_FreeSurface( bad_car2);
  SDL_FreeSurface( message );

  //Close the font that was used
  TTF_CloseFont( font );

  //Quit SDL
  SDL_Quit();
}

void DrivingWorld::initializeRoadPositions()
{
 //Set the wall
  right_wall.x = 5*LANE_WIDTH;
  right_wall.y = 0;
  right_wall.w = SCREEN_HEIGHT/40;
  right_wall.h = SCREEN_HEIGHT;

  left_wall.x = 0;
  left_wall.y = 0;
  left_wall.w = SCREEN_HEIGHT/40;
  left_wall.h = SCREEN_HEIGHT;

  //road for three lanes
  road.x = LANE_WIDTH;
  road.y = 0;
  road.w = 3 * LANE_WIDTH;
  road.h = SCREEN_HEIGHT;

  //offroad lanes
  offroad_left.x = 0;
  offroad_left.y = 0;
  offroad_left.w = LANE_WIDTH;
  offroad_left.h = SCREEN_HEIGHT;

  offroad_right.x = 4 * LANE_WIDTH;
  offroad_right.y = 0;
  offroad_right.w = LANE_WIDTH;
  offroad_right.h = SCREEN_HEIGHT;



  //lines for lanes
  line1.x = LANE_WIDTH + LANE_WIDTH;
  line1.y = 0;
  line1.w = 2;
  line1.h = SCREEN_HEIGHT;
  line2.x = 3 * LANE_WIDTH;
  line2.y = 0;
  line2.w = 2;
  line2.h = SCREEN_HEIGHT;

  //lines for offroad
  offroad_line1.x = LANE_WIDTH;
  offroad_line1.y = 0;
  offroad_line1.w = 5;
  offroad_line1.h = SCREEN_HEIGHT;
  offroad_line2.x = 4 * LANE_WIDTH;
  offroad_line2.y = 0;
  offroad_line2.w = 5;
  offroad_line2.h = SCREEN_HEIGHT;


}



#endif


