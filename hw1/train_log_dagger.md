# Ant

python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \

Collecting data for eval...
Eval_AverageReturn : 4779.64208984375
Eval_StdReturn : 80.10877227783203
Eval_MaxReturn : 4964.244140625
Eval_MinReturn : 4651.8671875
Eval_AverageEpLen : 1000.0
Train_AverageReturn : 4681.61279296875
Train_StdReturn : 596.7256469726562
Train_MaxReturn : 4980.9873046875
Train_MinReturn : 850.9080200195312
Train_AverageEpLen : 979.3009708737864
Train_EnvstepsSoFar : 903639
TimeSinceStart : 4681.247663497925
Training Loss : 0.0011522960849106312
Initial_DataCollection_AverageReturn : 4713.6533203125
Done logging...

# Hopper

python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Hopper.pkl \
    --env_name Hopper-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
    --video_log_freq -1