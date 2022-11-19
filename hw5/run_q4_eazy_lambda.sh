python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=0.1
--exp_name q4_awac_easy_supervised_lam0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=1
--exp_name q4_awac_easy_supervised_lam1

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=2
--exp_name q4_awac_easy_supervised_lam2

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=10
--exp_name q4_awac_easy_supervised_lam10

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=20
--exp_name q4_awac_easy_supervised_lam20

python cs285/scripts/run_hw5_awac.py --env_name PointmassEasy-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=50
--exp_name q4_awac_easy_supervised_lam50