# original one Ant-v4
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Ant.pkl \
	--env_name Ant-v4 --exp_name bc_ant --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
	--video_log_freq -1


Collecting data for eval...
Eval_AverageReturn : 4945.5615234375
Eval_StdReturn : 0.0
Eval_MaxReturn : 4945.5615234375
Eval_MinReturn : 4945.5615234375
Eval_AverageEpLen : 1000.0
Train_AverageReturn : 4713.6533203125
Train_StdReturn : 12.196533203125
Train_MaxReturn : 4725.849609375
Train_MinReturn : 4701.45654296875
Train_AverageEpLen : 1000.0
Train_EnvstepsSoFar : 0
TimeSinceStart : 2.9804418087005615
Training Loss : 0.0017621113220229745
Initial_DataCollection_AverageReturn : 4713.6533203125
Done logging...

# Walker 2d
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Walker2d.pkl \
	--env_name Walker2d-v4 --exp_name bc_Walker2d --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
	--video_log_freq -1

Collecting data for eval...
Eval_AverageReturn : 40.937679290771484
Eval_StdReturn : 85.34552764892578
Eval_MaxReturn : 289.45050048828125
Eval_MinReturn : 1.1642379760742188
Eval_AverageEpLen : 30.764705882352942
Train_AverageReturn : 5566.845703125
Train_StdReturn : 9.237548828125
Train_MaxReturn : 5576.08349609375
Train_MinReturn : 5557.6083984375
Train_AverageEpLen : 1000.0
Train_EnvstepsSoFar : 0
TimeSinceStart : 2.759728193283081
Training Loss : 0.0176960751414299
Initial_DataCollection_AverageReturn : 5566.845703125
Done logging...

# HalfCheetah
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/HalfCheetah.pkl \
	--env_name HalfCheetah-v4 --exp_name bc_HalfCheetah --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_HalfCheetah-v4.pkl \
	--video_log_freq -1

Collecting data for eval...
Eval_AverageReturn : 4143.2529296875
Eval_StdReturn : 0.0
Eval_MaxReturn : 4143.2529296875
Eval_MinReturn : 4143.2529296875
Eval_AverageEpLen : 1000.0
Train_AverageReturn : 4205.7783203125
Train_StdReturn : 83.038818359375
Train_MaxReturn : 4288.81689453125
Train_MinReturn : 4122.7392578125
Train_AverageEpLen : 1000.0
Train_EnvstepsSoFar : 0
TimeSinceStart : 2.805448532104492
Training Loss : 0.00423469627276063
Initial_DataCollection

# Hopper

python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Hopper.pkl \
	--env_name Hopper-v4 --exp_name bc_Hopper --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
	--video_log_freq -1

Collecting data for eval...
Eval_AverageReturn : 882.491455078125
Eval_StdReturn : 9.212494850158691
Eval_MaxReturn : 893.6419677734375
Eval_MinReturn : 872.4195556640625
Eval_AverageEpLen : 259.25
Train_AverageReturn : 3772.67041015625
Train_StdReturn : 1.9483642578125
Train_MaxReturn : 3774.61865234375
Train_MinReturn : 3770.721923828125
Train_AverageEpLen : 1000.0
Train_EnvstepsSoFar : 0
TimeSinceStart : 3.008108139038086
Training Loss : 0.008262570016086102
Initial_DataCollection_AverageReturn : 3772.67041015625
Done logging...





