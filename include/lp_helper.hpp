#ifndef lp_helper_h
#define lp_helper_h

#include <stdio.h>
#include <stdlib.h>
#include "lp_lib.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace std;


//solve 
//min c^t x
//s.t. Ax >= b
//Need to remember to pad front of a's and c with a zero
double solve_lp(vector<double> c, vector<vector<double> > A, vector<double> b)
{
 
    //unsigned int n_rows = A.size();
    unsigned int n_cols = c.size();
    

    
    lprec *lp;
    
    //cols index starting at 1
    vector<int> colno(A[0].size());
    for(unsigned int i=0; i<n_cols;i++)
        colno[i] = i + 1;

    /* Create a new LP model */
    //TODO: figure out how to use preallocated rows
    lp = make_lp(0, n_cols);
    if(lp == NULL) {
        fprintf(stderr, "Unable to create new LP model\n");
        return(1);
    }
    //set variables as unbounded (-inf, +inf)
    for(unsigned int col = 1; col <= n_cols; col++)
        set_unbounded(lp, col);
    
    //set objective function
    //set_maxim(lp);
    set_obj_fnex(lp, n_cols, &c[0], &colno[0]);
    
    //add rows for A x <= b
    set_add_rowmode(lp, TRUE);
    


    //iterate over each constraint
    for(unsigned int i=0; i<A.size(); i++)
    {
        REAL row_vals[n_cols];
        for(unsigned int j=0;j<n_cols;j++)
            row_vals[j] = A[i][j];
        add_constraintex(lp, n_cols, row_vals, &colno[0],  GE, b[i]);
        //add_constraint(lp, &A[i][0], GE, b[i]);
    }
    
    set_add_rowmode(lp, FALSE);
    
    //for debugging printout the lp
    //write_LP(lp, stdout);
    
    set_scaling(lp, SCALE_MEAN | SCALE_LOGARITHMIC);
    //set_maxpivot(lp, 100);
    //set_presolve(lp, PRESOLVE_ROWS | PRESOLVE_COLS | PRESOLVE_LINDEP, get_presolveloops(lp));
    
    // I only want to see important messages on screen while solving
    set_verbose(lp, NEUTRAL);//set to no messages
    set_timeout(lp, 1); /* sets a timeout of 2 seconds */
    //TODO: why does it hang sometimes? I had to add the timeout to kick it out from 
    //numerical instabilities, not sure why, maybe CPLEX would be better!

    // Now let lpsolve calculate a solution
    int ret = solve(lp);
    double obj_val = 0;
    if(ret == OPTIMAL)
    {
        /* objective value */
        obj_val = get_objective(lp);
        //printf("Objective value: %f\n", obj_val);

        /* variable values */
        //REAL var[n_cols];
        //get_variables(lp, var);
        //for(int j = 0; j < n_cols; j++)
        //  printf("%s: %f\n", get_col_name(lp, j + 1), var[j]);

    }
    else if(ret == UNBOUNDED)
    {
        //cout << "return value: " << ret << endl;
        //fprintf(stderr, "Unbounded\n");
        obj_val = -9999;  //TODO maybe a better way, but this should show it is unbounded
    }
    else if(ret == SUBOPTIMAL)
    {
        cout << "Time out!!!!!!!!!" << endl;
        cout << obj_val << endl;
    }
    else
    {
         cout << "return value: " << ret << endl;
         fprintf(stderr, "Problem solving LP\n");
    }
    
    delete_lp(lp);
    return obj_val;
}

bool areVectorsEqual(vector<double> a, vector<double> b)
{
    if(a.size() != b.size())
        return false;
    for(unsigned int i=0; i < a.size(); i++)
        if(abs(a[i] - b[i]) > 0.00001)
            return false;
    return true;
}

//check if any rows of A are redundant for constraints
// Ax >= 0
vector<vector<double> > old_removeRedundantHalfspaces(vector<vector<double> > A)
{
    vector<vector<double> > non_redundant;
    //check each row for redundancy by making it the objective
    for(vector<double> a : A)
    {
        //keep the other constraints
        vector<vector<double> > A_other;
        for(vector<double> b : A)
        {
            if(!areVectorsEqual(a,b))
            {
                A_other.push_back(b);
            }
        }
        //solve LP with a as objective
        vector<double> b(A_other.size());
        for(unsigned int i=0;i<A_other.size();i++)
            b[i] = 0.0;
        double obj_val = solve_lp(a, A_other,b);
        //cout << "Objective return value = " << obj_val << endl;
        
        //need to check for unboundedness too! : I currently return -9999 if unbounded
        //if unbounded then need to keep it.
        //if obj_val < 0 then we need to keep it
        if(obj_val < 0)
            non_redundant.push_back(a);
    }
    return non_redundant;
}

//check if any rows of A are redundant for constraints
// Ax >= 0
//TODO make this use a set or other data structure for easy removal
vector<vector<double> > removeRedundantHalfspaces(vector<vector<double> > A)
{
    vector<vector<double> > non_redundant; //local copy
    for(vector<double> a : A)
    {
        vector<double> a_copy = a;
        non_redundant.push_back(a_copy);
    }
    //check each row for redundancy by making it the objective
    for(vector<double> a : A)
    {
        //keep the other constraints
        vector<vector<double> > A_other;
        for(vector<double> b : non_redundant)
        {
            if(!areVectorsEqual(a,b))
            {
                A_other.push_back(b);
            }
        }
        //solve LP with a as objective
        vector<double> b(A_other.size());
        for(unsigned int i=0;i<A_other.size();i++)
            b[i] = 0.0;
        double obj_val = solve_lp(a, A_other,b);
        //cout << "Objective return value = " << obj_val << endl;
        
        //need to check for unboundedness too! : I currently return -9999 if unbounded
        //if unbounded then need to keep it.
        //if obj_val < 0 then we need to keep it
        if(obj_val >= 0)
        {
            vector<vector<double> > copy;
            for(vector<double> c : non_redundant)
                if(!areVectorsEqual(a,c))
                    copy.push_back(c);
            //remove a from non_redundant
            non_redundant = copy;
        }
    }
    return non_redundant;
}


//check if any elements in objConstraints are redundant for constraints
// Ax >= 0
//TODO make this use a set or other data structure for easy removal
vector<vector<double> > removeRedundantHalfspaces(vector<vector<double> > objConstraints, vector<vector<double> > A)
{
    vector<vector<double> > non_redundant; //local copy
    for(vector<double> a : objConstraints)
    {
        vector<double> a_copy = a;
        non_redundant.push_back(a_copy);
    }
    //check each row for redundancy by making it the objective
    for(vector<double> a : objConstraints)
    {

        //solve LP with a as objective
        vector<double> b(A.size());
        for(unsigned int i=0;i<A.size();i++)
            b[i] = 0.0;
        double obj_val = solve_lp(a, A, b);
        //cout << "Objective return value = " << obj_val << endl;
        
        //need to check for unboundedness too! : I currently return -9999 if unbounded
        //if unbounded then need to keep it.
        //if obj_val < 0 then we need to keep it
        if(obj_val >= 0)
        {
            vector<vector<double> > copy;
            for(vector<double> c : non_redundant)
                if(!areVectorsEqual(a,c))
                    copy.push_back(c);
            //remove a from non_redundant
            non_redundant = copy;
        }
    }
    return non_redundant;
}

#endif
