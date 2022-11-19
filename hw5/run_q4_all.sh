python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam0.1 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=0.1

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=1 \
--exp_name q4_awac_medium_supervised_lam1

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam2 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=2

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=2 \
--exp_name q4_awac_medium_supervised_lam2

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam10 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=10

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=10 \
--exp_name q4_awac_medium_supervised_lam10

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam20 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=20

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=20 \
--exp_name q4_awac_medium_supervised_lam20

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 \
--exp_name q4_awac_medium_unsupervised_lam50 --use_rnd --num_exploration_steps=20000 \
--unsupervised_exploration --awac_lambda=50

python cs285/scripts/run_hw5_awac.py --env_name PointmassMedium-v0 --use_rnd \
--num_exploration_steps=20000 --awac_lambda=50 \
--exp_name q4_awac_medium_supervised_lam50