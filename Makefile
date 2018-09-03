OBJS = build/birl_test.o build/feature_test.o build/mdp_test.o
CC = g++ -std=c++11
DEBUG = -g -O3 -fopenmp
CFLAGS = -Wall -c $(DEBUG)
LFLAGS = -Wall $(DEBUG)

LP_SOLVE_INCL = -I /home/dsbrown/Libraries/lpsolve55/ -I /home/dsbrown/Libraries/eigen
LP_SOLVE_LINK = /home/dsbrown/Libraries/lpsolve55/liblpsolve55.a -ldl



all: birl_test feature_test mdp_test feature_birl_test conf_bound_test grid_experiment conf_bound_feature_test grid_l1_experiment fcount_example driving_example driving_birl_example experiment1 experiment2 experiment3 experiment4 experiment5 experiment7 experiment8 driving_experiment1 driving_birl_sandbox experiment9 experiment10 poison_test teaching1 enough1 enough2 enough_toy improvement_toy gen_feasible_r cakmak_test

feature: feature_test mdp_test feature_birl_test

feature_test: build/feature_test.o
	$(CC) $(LFLAGS) build/feature_test.o -o feature_test 
	
feature_birl_test: build/feature_birl_test.o
	$(CC) $(LFLAGS) build/feature_birl_test.o -o feature_birl_test 
	
poison_test: build/poison_test.o
	$(CC) $(LFLAGS) build/poison_test.o -o poison_test 

birl_test: $(OBJS)
	$(CC) $(LFLAGS) build/birl_test.o -o birl_test
	
chain_test: build/chain_test.o
	$(CC) $(LFLAGS) build/chain_test.o -o chain_test $(LP_SOLVE_LINK)	
	
bandit_placement: build/bandit_placement_test.o    
	$(CC) $(LFLAGS) build/bandit_placement_test.o -o bandit_placement $(LP_SOLVE_LINK)	
	
chain_grid_search: build/chain_grid_search.o
	$(CC) $(LFLAGS) build/chain_grid_search.o -o chain_grid_search $(LP_SOLVE_LINK)	

bioirl_exp1: build/bioirl_experiment1.o
	$(CC) $(LFLAGS) build/bioirl_experiment1.o -o bioirl_exp1 $(LP_SOLVE_LINK)	

bioirl_exp2: build/bioirl_experiment2.o
	$(CC) $(LFLAGS) build/bioirl_experiment2.o -o bioirl_exp2 $(LP_SOLVE_LINK)	

bioirl_exp3: build/bioirl_experiment3.o
	$(CC) $(LFLAGS) build/bioirl_experiment3.o -o bioirl_exp3 $(LP_SOLVE_LINK)	

bioirl_exp4: build/bioirl_experiment4.o
	$(CC) $(LFLAGS) build/bioirl_experiment4.o -o bioirl_exp4 $(LP_SOLVE_LINK)	

bioirl_exp5: build/bioirl_experiment5.o
	$(CC) $(LFLAGS) build/bioirl_experiment5.o -o bioirl_exp5 $(LP_SOLVE_LINK)	

bioirl_exp6: build/bioirl_experiment6.o
	$(CC) $(LFLAGS) build/bioirl_experiment6.o -o bioirl_exp6 $(LP_SOLVE_LINK)	

bioirl_exp7: build/bioirl_experiment7.o
	$(CC) $(LFLAGS) build/bioirl_experiment7.o -o bioirl_exp7 $(LP_SOLVE_LINK)	
	
mdp_test: build/mdp_test.o
	$(CC) $(LFLAGS) build/mdp_test.o -o mdp_test	

conf_bound_test: build/conf_bound_test.o 
	$(CC) $(LFLAGS) build/conf_bound_test.o -o conf_bound_test	
	
conf_bound_feature_test: build/conf_bound_feature_test.o
	$(CC) $(LFLAGS) build/conf_bound_feature_test.o -o conf_bound_feature_test	

grid_experiment: build/gridNavExperimentRunner.o
	$(CC) $(LFLAGS) build/gridNavExperimentRunner.o -o grid_experiment	

grid_l1_experiment: build/gridNavExperimentRunnerL1walk.o
	$(CC) $(LFLAGS) build/gridNavExperimentRunnerL1walk.o -o grid_l1_experiment	

fcount_example: build/gridNavFeatureCntExampleL1walk.o
	$(CC) $(LFLAGS) build/gridNavFeatureCntExampleL1walk.o -o fcount_example
	
driving_example: build/drivingTester.o
	$(CC) $(LFLAGS) build/drivingTester.o -o driving_example -I /usr/include/SDL -lSDL -lSDL_image -lpthread -lSDL_ttf

driving_birl_example: build/drivingBIRLTester.o
	$(CC) $(LFLAGS) build/drivingBIRLTester.o  -I /usr/include/SDL -lSDL -lSDL_image -lpthread -lSDL_ttf -o driving_birl_example
	
driving_birl_demo: build/drivingBIRL_Demo.o
	$(CC) $(LFLAGS) build/drivingBIRL_Demo.o  -I /usr/include/SDL -lSDL -lSDL_image -lpthread -lSDL_ttf -o driving_birl_demo

driving_birl_replay: build/drivingBIRL_Replay.o
	$(CC) $(LFLAGS) build/drivingBIRL_Replay.o  -I /usr/include/SDL -lSDL -lSDL_image -lpthread -lSDL_ttf -o driving_birl_replay

driving_birl_sandbox: build/drivingBIRLSandbox.o
	$(CC) $(LFLAGS) build/drivingBIRLSandbox.o  -I /usr/include/SDL -lSDL -lSDL_image -lpthread -lSDL_ttf -o driving_birl_sandbox

	
driving_experiment1: build/drivingBIRLExperiment1.o
	$(CC) $(LFLAGS) build/drivingBIRLExperiment1.o  -I /usr/include/SDL -lSDL -lSDL_image -lpthread -lSDL_ttf -o driving_experiment1	

driving_experiment2: build/drivingBIRLExperiment2.o
	$(CC) $(LFLAGS) build/drivingBIRLExperiment2.o  -I /usr/include/SDL -lSDL -lSDL_image -lpthread -lSDL_ttf -o driving_experiment2	

driving_experiment3: build/drivingBIRLExperiment3.o
	$(CC) $(LFLAGS) build/drivingBIRLExperiment3.o  -I /usr/include/SDL -lSDL -lSDL_image -lpthread -lSDL_ttf -o driving_experiment3	

driving_ccounts: build/drivingBIRLCollisionCounts.o
	$(CC) $(LFLAGS) build/drivingBIRLCollisionCounts.o  -I /usr/include/SDL -lSDL -lSDL_image -lpthread -lSDL_ttf -o driving_ccounts	


experiment1: build/gridWorldExperiment1.o
	$(CC) $(LFLAGS) build/gridWorldExperiment1.o  -o experiment1
	
experiment2: build/gridWorldExperiment2.o
	$(CC) $(LFLAGS) build/gridWorldExperiment2.o  -o experiment2

experiment3: build/gridWorldExperiment3.o
	$(CC) $(LFLAGS) build/gridWorldExperiment3.o  -o experiment3

experiment4: build/gridWorldExperiment4.o
	$(CC) $(LFLAGS) build/gridWorldExperiment4.o  -o experiment4

experiment5: build/gridWorldExperiment5.o
	$(CC) $(LFLAGS) build/gridWorldExperiment5.o  -o experiment5
	
experiment7: build/gridWorldExperiment7.o
	$(CC) $(LFLAGS) build/gridWorldExperiment7.o  -o experiment7

experiment8: build/gridWorldExperiment8.o
	$(CC) $(LFLAGS) build/gridWorldExperiment8.o  -o experiment8

experiment9: build/gridWorldExperiment9.o
	$(CC) $(LFLAGS) build/gridWorldExperiment9.o  -o experiment9

experiment10: build/gridWorldExperiment10.o
	$(CC) $(LFLAGS) build/gridWorldExperiment10.o  -o experiment10

enough1: build/gridWorldExperimentEnoughEnough1.o
	$(CC) $(LFLAGS) build/gridWorldExperimentEnoughEnough1.o  -o enough1

enough2: build/gridWorldExperimentEnoughEnough2.o
	$(CC) $(LFLAGS) -pg build/gridWorldExperimentEnoughEnough2.o  -o enough2

enough_toy: build/enoughToyExample.o
	$(CC) $(LFLAGS) -pg build/enoughToyExample.o  -o enough_toy
	
gen_feasible_r: build/enoughToyExample_gen_all_feasible_rewards.o
	$(CC) $(LFLAGS) -pg build/enoughToyExample_gen_all_feasible_rewards.o  -o gen_feasible_r
	
improvement_toy: build/improvementToyExample.o
	$(CC) $(LFLAGS) -pg build/improvementToyExample.o  -o improvement_toy
	
active_VaR_toy: build/activeVaRToyExample.o
	$(CC) $(LFLAGS) -pg build/activeVaRToyExample.o  -o active_VaR_toy

teaching1: build/optimalTeachingExperiment1.o
	$(CC) $(LFLAGS) build/optimalTeachingExperiment1.o  -o teaching1
	
cakmak_test: build/cakmakTest.o
	$(CC) $(LFLAGS) build/cakmakTest.o  -o cakmak_test $(LP_SOLVE_LINK)

min_demo_experiment: build/minDemosForOptimal.o
	$(CC) $(LFLAGS) build/minDemosForOptimal.o  -o min_demo_experiment $(LP_SOLVE_LINK)
	
min_demo_experiment_sc: build/minDemosForOptimal_SetCover.o
	$(CC) $(LFLAGS) build/minDemosForOptimal_SetCover.o  -o min_demo_experiment_sc $(LP_SOLVE_LINK)


teaching_irl_test: build/teachingAwareIRL.o
	$(CC) $(LFLAGS) build/teachingAwareIRL.o  -o teaching_irl_test $(LP_SOLVE_LINK)
	
cakmak_task_experiment: build/cakmakTasksExperiment.o
	$(CC) $(LFLAGS) build/cakmakTasksExperiment.o  -o cakmak_task_experiment $(LP_SOLVE_LINK)
	
active_bench: build/activeLearningBaseline.o
	$(CC) $(LFLAGS) build/activeLearningBaseline.o  -o active_bench $(LP_SOLVE_LINK)
	
active_test_all: build/active_benchmark_test_all.o
	$(CC) $(LFLAGS) build/active_benchmark_test_all.o  -o active_test_all $(LP_SOLVE_LINK)

active_critique: build/active_critique_vs_random.o
	$(CC) $(LFLAGS) build/active_critique_vs_random.o  -o active_critique

cakmak_sensitivity_experiment: build/cakmakRandomSensitivityExperiment.o
	$(CC) $(LFLAGS) build/cakmakRandomSensitivityExperiment.o  -o cakmak_sensitivity_experiment $(LP_SOLVE_LINK)
	
opt_feature_sensitivity_experiment: build/cakmakRandomFeatureSensitivityExperiment.o
	$(CC) $(LFLAGS) build/cakmakRandomFeatureSensitivityExperiment.o  -o opt_feature_sensitivity_experiment $(LP_SOLVE_LINK)
	
algo_comp: build/machineTeachingAlgorithmComparison.o
	$(CC) $(LFLAGS) build/machineTeachingAlgorithmComparison.o  -o algo_comp $(LP_SOLVE_LINK)
	
build/chain_test.o: src/chain_test.cpp include/mdp.hpp include/feature_chain_birl.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp
	$(CC) $(CFLAGS) src/chain_test.cpp -o build/chain_test.o $(LP_SOLVE_INCL)
	
build/bandit_placement_test.o: src/bandit_placement_test.cpp include/mdp.hpp include/bandit.hpp include/optimalTeaching.hpp include/feature_bandit_birl.hpp include/confidence_bounds.hpp include/feature_gain.hpp
	$(CC) $(CFLAGS) src/bandit_placement_test.cpp -o build/bandit_placement_test.o $(LP_SOLVE_INCL)

build/bioirl_experiment1.o: src/bioirl_experiment1.cpp include/mdp.hpp include/maxent_feature_birl.hpp include/optimalTeaching.hpp
	$(CC) $(CFLAGS) src/bioirl_experiment1.cpp -o build/bioirl_experiment1.o $(LP_SOLVE_INCL)

build/bioirl_experiment2.o: src/bioirl_experiment2.cpp include/mdp.hpp include/maxent_feature_birl.hpp include/optimalTeaching.hpp
	$(CC) $(CFLAGS) src/bioirl_experiment2.cpp -o build/bioirl_experiment2.o $(LP_SOLVE_INCL)

build/bioirl_experiment3.o: src/bioirl_experiment3.cpp include/mdp.hpp include/maxent_feature_birl.hpp include/optimalTeaching.hpp
	$(CC) $(CFLAGS) src/bioirl_experiment3.cpp -o build/bioirl_experiment3.o $(LP_SOLVE_INCL)

build/bioirl_experiment4.o: src/bioirl_experiment4.cpp include/mdp.hpp include/maxent_feature_birl.hpp include/optimalTeaching.hpp
	$(CC) $(CFLAGS) src/bioirl_experiment4.cpp -o build/bioirl_experiment4.o $(LP_SOLVE_INCL)

build/bioirl_experiment5.o: src/bioirl_experiment5.cpp include/mdp.hpp include/maxent_feature_birl.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp
	$(CC) $(CFLAGS) src/bioirl_experiment5.cpp -o build/bioirl_experiment5.o $(LP_SOLVE_INCL)

build/bioirl_experiment6.o: src/bioirl_experiment6.cpp include/mdp.hpp include/maxent_feature_birl.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/grid_domains.hpp
	$(CC) $(CFLAGS) src/bioirl_experiment6.cpp -o build/bioirl_experiment6.o $(LP_SOLVE_INCL)

build/bioirl_experiment7.o: src/bioirl_experiment7.cpp include/mdp.hpp include/maxent_feature_birl.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/grid_domains.hpp
	$(CC) $(CFLAGS) src/bioirl_experiment7.cpp -o build/bioirl_experiment7.o $(LP_SOLVE_INCL)

build/chain_grid_search.o: src/chain_grid_search.cpp include/mdp.hpp include/feature_chain_birl.hpp include/optimalTeaching.hpp
	$(CC) $(CFLAGS) src/chain_grid_search.cpp -o build/chain_grid_search.o $(LP_SOLVE_INCL)

build/drivingTester.o: src/drivingTester.cpp include/q_learner_driving.hpp include/driving_world.hpp 
	$(CC) $(CFLAGS) src/drivingTester.cpp -o build/drivingTester.o 

build/drivingBIRLTester.o: src/drivingBIRLTester.cpp include/q_learner_driving.hpp include/driving_world.hpp include/feature_birl_qlearning.hpp include/confidence_bounds_qlearning.hpp
	$(CC) $(CFLAGS) src/drivingBIRLTester.cpp -o build/drivingBIRLTester.o 
	
build/drivingBIRL_Demo.o: src/drivingBIRL_Demo.cpp include/q_learner_driving_demo.hpp include/driving_world_demo.hpp include/feature_birl_qlearning_demo.hpp include/confidence_bounds_qlearning_demo.hpp
	$(CC) $(CFLAGS) src/drivingBIRL_Demo.cpp -o build/drivingBIRL_Demo.o 

build/drivingBIRL_Replay.o: src/drivingBIRL_Replay.cpp include/q_learner_driving_demo.hpp include/driving_world_demo.hpp include/feature_birl_qlearning_demo.hpp include/confidence_bounds_qlearning_demo.hpp
	$(CC) $(CFLAGS) src/drivingBIRL_Replay.cpp -o build/drivingBIRL_Replay.o 
	
build/drivingBIRLSandbox.o: src/drivingBIRLSandbox.cpp include/q_learner_driving.hpp include/driving_world.hpp include/feature_birl_qlearning.hpp include/confidence_bounds_qlearning.hpp
	$(CC) $(CFLAGS) src/drivingBIRLSandbox.cpp -o build/drivingBIRLSandbox.o 
	
build/drivingBIRLExperiment1.o: src/drivingBIRLExperiment1.cpp include/q_learner_driving.hpp include/driving_world.hpp include/feature_birl_qlearning.hpp include/unit_norm_sampling.hpp include/confidence_bounds_qlearning.hpp
	$(CC) $(CFLAGS) src/drivingBIRLExperiment1.cpp -o build/drivingBIRLExperiment1.o 
	
	
build/drivingBIRLExperiment2.o: src/drivingBIRLExperiment2.cpp include/q_learner_driving.hpp include/driving_world.hpp include/feature_birl_qlearning.hpp include/unit_norm_sampling.hpp include/confidence_bounds_qlearning.hpp
	$(CC) $(CFLAGS) src/drivingBIRLExperiment2.cpp -o build/drivingBIRLExperiment2.o 

build/drivingBIRLExperiment3.o: src/drivingBIRLExperiment3.cpp include/q_learner_driving.hpp include/driving_world.hpp include/feature_birl_qlearning.hpp include/unit_norm_sampling.hpp include/confidence_bounds_qlearning.hpp
	$(CC) $(CFLAGS) src/drivingBIRLExperiment3.cpp -o build/drivingBIRLExperiment3.o 

build/drivingBIRLCollisionCounts.o: src/drivingBIRLCollisionCounts.cpp include/q_learner_driving.hpp include/driving_world.hpp include/feature_birl_qlearning.hpp include/unit_norm_sampling.hpp include/confidence_bounds_qlearning.hpp
	$(CC) $(CFLAGS) src/drivingBIRLCollisionCounts.cpp -o build/drivingBIRLCollisionCounts.o 	

build/birl_test: $(OBJS)
	$(CC) $(LFLAGS) $(OBJS) -o birl_test

build/birl_test.o: src/birl_test.cpp include/birl.hpp include/mdp.hpp
	$(CC) $(CFLAGS) src/birl_test.cpp -o build/birl_test.o
	
build/feature_test.o: src/feature_test.cpp include/mdp.hpp include/grid_domains.hpp include/confidence_bounds.hpp include/abbeel_projection.hpp
	$(CC) $(CFLAGS) src/feature_test.cpp -o build/feature_test.o

build/feature_birl_test.o: src/feature_birl_test.cpp include/feature_birl.hpp include/mdp.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp include/abbeel_projection.hpp
	$(CC) $(CFLAGS) src/feature_birl_test.cpp -o build/feature_birl_test.o
	
build/poison_test.o: src/poison_test.cpp include/feature_birl.hpp include/mdp.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp include/abbeel_projection.hpp
	$(CC) $(CFLAGS) src/poison_test.cpp -o build/poison_test.o

build/mdp_test.o: src/mdp_test.cpp include/mdp.hpp include/q_learner_grid.hpp
	$(CC) $(CFLAGS) src/mdp_test.cpp -o build/mdp_test.o

build/conf_bound_test.o: src/conf_bound_test.cpp include/mdp.hpp include/confidence_bounds.hpp
	$(CC) $(CFLAGS) src/conf_bound_test.cpp -o build/conf_bound_test.o

build/conf_bound_feature_test.o: src/conf_bound_feature_test.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/mdp.hpp include/grid_domains.hpp
	$(CC) $(CFLAGS) src/conf_bound_feature_test.cpp -o build/conf_bound_feature_test.o


build/gridNavExperimentRunner.o: src/gridNavExperimentRunner.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridNavExperimentRunner.cpp -o build/gridNavExperimentRunner.o

build/gridNavExperimentRunnerL1walk.o: src/gridNavExperimentRunnerL1walk.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridNavExperimentRunnerL1walk.cpp -o build/gridNavExperimentRunnerL1walk.o

build/gridNavFeatureCntExampleL1walk.o: src/gridNavFeatureCntExampleL1walk.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridNavFeatureCntExampleL1walk.cpp -o build/gridNavFeatureCntExampleL1walk.o

build/gridWorldExperiment1.o: src/gridWorldExperiment1.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridWorldExperiment1.cpp -o build/gridWorldExperiment1.o

build/gridWorldExperiment2.o: src/gridWorldExperiment2.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridWorldExperiment2.cpp -o build/gridWorldExperiment2.o

build/gridWorldExperiment3.o: src/gridWorldExperiment3.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridWorldExperiment3.cpp -o build/gridWorldExperiment3.o

build/gridWorldExperiment4.o: src/gridWorldExperiment4.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridWorldExperiment4.cpp -o build/gridWorldExperiment4.o

build/gridWorldExperiment5.o: src/gridWorldExperiment5.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridWorldExperiment5.cpp -o build/gridWorldExperiment5.o

build/gridWorldExperiment7.o: src/gridWorldExperiment7.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridWorldExperiment7.cpp -o build/gridWorldExperiment7.o

build/gridWorldExperiment8.o: src/gridWorldExperiment8.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridWorldExperiment8.cpp -o build/gridWorldExperiment8.o


build/gridWorldExperiment9.o: src/gridWorldExperiment9.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridWorldExperiment9.cpp -o build/gridWorldExperiment9.o


build/gridWorldExperiment10.o: src/gridWorldExperiment10.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp include/abbeel_projection.hpp
	$(CC) $(CFLAGS) src/gridWorldExperiment10.cpp -o build/gridWorldExperiment10.o


build/gridWorldExperimentEnoughEnough1.o: src/gridWorldExperimentEnoughEnough1.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/gridWorldExperimentEnoughEnough1.cpp -o build/gridWorldExperimentEnoughEnough1.o

build/gridWorldExperimentEnoughEnough2.o: src/gridWorldExperimentEnoughEnough2.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) -pg src/gridWorldExperimentEnoughEnough2.cpp -o build/gridWorldExperimentEnoughEnough2.o

build/enoughToyExample.o: src/enoughToyExample.cpp include/mdp.hpp include/confidence_bounds.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp include/maxent_feature_birl.hpp
	$(CC) $(CFLAGS) -pg src/enoughToyExample.cpp -o build/enoughToyExample.o

build/enoughToyExample_gen_all_feasible_rewards.o: src/enoughToyExample_gen_all_feasible_rewards.cpp include/mdp.hpp include/confidence_bounds.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp include/maxent_feature_birl.hpp
	$(CC) $(CFLAGS) -pg src/enoughToyExample_gen_all_feasible_rewards.cpp -o build/enoughToyExample_gen_all_feasible_rewards.o


build/improvementToyExample.o: src/improvementToyExample.cpp include/mdp.hpp include/confidence_bounds.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp include/maxent_feature_birl.hpp
	$(CC) $(CFLAGS) -pg src/improvementToyExample.cpp -o build/improvementToyExample.o
	
build/activeVaRToyExample.o: src/activeVaRToyExample.cpp include/mdp.hpp include/confidence_bounds.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp include/maxent_feature_birl.hpp
	$(CC) $(CFLAGS) -pg src/activeVaRToyExample.cpp -o build/activeVaRToyExample.o


build/optimalTeachingExperiment1.o: src/optimalTeachingExperiment1.cpp include/mdp.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/grid_domains.hpp include/unit_norm_sampling.hpp include/machine_teaching.hpp
	$(CC) $(CFLAGS) src/optimalTeachingExperiment1.cpp -o build/optimalTeachingExperiment1.o

build/cakmakTest.o: src/cakmakTest.cpp include/mdp.hpp include/grid_domains.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/lp_helper.hpp include/maxent_feature_birl.hpp include/feature_birl.hpp
	$(CC) $(CFLAGS) src/cakmakTest.cpp -o build/cakmakTest.o $(LP_SOLVE_INCL)

build/teachingAwareIRL.o: src/teachingAwareIRL.cpp include/mdp.hpp include/grid_domains.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/lp_helper.hpp include/maxent_feature_birl.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/teachingAwareIRL.cpp -o build/teachingAwareIRL.o $(LP_SOLVE_INCL)
	
build/cakmakTasksExperiment.o: src/cakmakTasksExperiment.cpp include/mdp.hpp include/grid_domains.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/lp_helper.hpp include/feature_birl.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/cakmakTasksExperiment.cpp -o build/cakmakTasksExperiment.o $(LP_SOLVE_INCL)
	
build/activeLearningBaseline.o: src/activeLearningBaseline.cpp include/mdp.hpp include/grid_domains.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/lp_helper.hpp include/feature_birl.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/activeLearningBaseline.cpp -o build/activeLearningBaseline.o $(LP_SOLVE_INCL)
	
build/active_benchmark_test_all.o: src/active_benchmark_test_all.cpp include/mdp.hpp include/grid_domains.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/unit_norm_sampling.hpp include/optimalTeaching.hpp include/feature_gain_info.hpp
	$(CC) $(CFLAGS) src/active_benchmark_test_all.cpp -o build/active_benchmark_test_all.o $(LP_SOLVE_INCL)

build/active_critique_vs_random.o: src/active_critique_vs_random.cpp include/mdp.hpp include/grid_domains.hpp include/confidence_bounds.hpp include/feature_birl.hpp include/unit_norm_sampling.hpp include/feature_gain_info.hpp
	$(CC) $(CFLAGS) src/active_critique_vs_random.cpp -o build/active_critique_vs_random.o
	
build/minDemosForOptimal.o: src/minDemosForOptimal.cpp include/mdp.hpp include/grid_domains.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/lp_helper.hpp include/feature_birl.hpp include/unit_norm_sampling.hpp 
	$(CC) $(CFLAGS) src/minDemosForOptimal.cpp -o build/minDemosForOptimal.o $(LP_SOLVE_INCL)
	
build/minDemosForOptimal_SetCover.o: src/minDemosForOptimal_SetCover.cpp include/mdp.hpp include/grid_domains.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/lp_helper.hpp include/feature_birl.hpp include/unit_norm_sampling.hpp 
	$(CC) $(CFLAGS) src/minDemosForOptimal_SetCover.cpp -o build/minDemosForOptimal_SetCover.o $(LP_SOLVE_INCL)
	
build/cakmakRandomSensitivityExperiment.o: src/cakmakRandomSensitivityExperiment.cpp include/mdp.hpp include/grid_domains.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/lp_helper.hpp include/feature_birl.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/cakmakRandomSensitivityExperiment.cpp -o build/cakmakRandomSensitivityExperiment.o $(LP_SOLVE_INCL)
	
build/cakmakRandomFeatureSensitivityExperiment.o: src/cakmakRandomFeatureSensitivityExperiment.cpp include/mdp.hpp include/grid_domains.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/lp_helper.hpp include/feature_birl.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/cakmakRandomFeatureSensitivityExperiment.cpp -o build/cakmakRandomFeatureSensitivityExperiment.o $(LP_SOLVE_INCL)
	
	
build/machineTeachingAlgorithmComparison.o: src/machineTeachingAlgorithmComparison.cpp include/mdp.hpp include/grid_domains.hpp include/optimalTeaching.hpp include/confidence_bounds.hpp include/lp_helper.hpp include/feature_birl.hpp include/unit_norm_sampling.hpp
	$(CC) $(CFLAGS) src/machineTeachingAlgorithmComparison.cpp -o build/machineTeachingAlgorithmComparison.o $(LP_SOLVE_INCL)

clean:
	\rm build/*.o src/*~ birl_test feature_test mdp_test feature_birl_test conf_bound_test grid_experiment conf_bound_feature_test grid_l1_experiment fcount_example driving_example driving_birl_example experiment1 experiment2 experiment3 experiment4 experiment5 experiment7 experiment8 driving_experiment1 driving_birl_sandbox experiment9 experiment10 poison_test teaching1 enough1 enough2 enough_toy
	    
memchk_mdp:
	valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=40 --track-fds=yes --track-origins=yes ./mdp_test
	
memchk_feature:
	valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=40 --track-fds=yes --track-origins=yes ./feature_test	

memchk_birl:
	valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=40 --track-fds=yes --track-origins=yes ./birl_test
	
memchk_fbirl:
	valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=40 --track-fds=yes --track-origins=yes ./feature_birl_test
	
memchk_expRunner:
	valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=40 --track-fds=yes --track-origins=yes ./grid_experiment_l1

memchk_gridEE:
	valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=40 --track-fds=yes --track-origins=yes ./enough2
	
memchk_opt_sens:
	valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=40 --track-fds=yes --track-origins=yes ./opt_feature_sensitivity_experiment non-redundant random_mixed 1
	
	
memchk_opt_sens_cak:
	valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=40 --track-fds=yes --track-origins=yes ./opt_feature_sensitivity_experiment cakmak random 1
	
memchk_opt_sens_scot:
	valgrind --tool=memcheck --leak-check=yes --show-reachable=yes --num-callers=40 --track-fds=yes --track-origins=yes ./opt_feature_sensitivity_experiment scot random_mixed 1

