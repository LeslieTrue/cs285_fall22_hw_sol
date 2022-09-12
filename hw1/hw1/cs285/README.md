## Commands for Q 1.3

```
python cs285/scripts/run_hw1.py \
	--expert_policy_file cs285/policies/experts/Walker2d.pkl \
	--env_name Walker2d-v4 --exp_name bc_Walker2d --n_iter 1 \
	--expert_data cs285/expert_data/expert_data_Walker2d-v4.pkl \
	--video_log_freq -1
    --num_agent_train_steps_per_iter 1000
```

Just change the bottom line with: 

--num_agent_train_steps_per_iter 1000

--num_agent_train_steps_per_iter 2000

--num_agent_train_steps_per_iter 5000

--num_agent_train_steps_per_iter 8000

--num_agent_train_steps_per_iter 10000

--num_agent_train_steps_per_iter 30000

--num_agent_train_steps_per_iter 50000

--num_agent_train_steps_per_iter 80000

--num_agent_train_steps_per_iter 100000

## Command for Q 2.2

### Ant:

```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Ant.pkl \
    --env_name Ant-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Ant-v4.pkl \
    --video_log_freq -1
```

### Hopper;

```
python cs285/scripts/run_hw1.py \
    --expert_policy_file cs285/policies/experts/Hopper.pkl \
    --env_name Hopper-v4 --exp_name dagger_ant --n_iter 10 \
    --do_dagger --expert_data cs285/expert_data/expert_data_Hopper-v4.pkl \
    --video_log_freq -1
```

