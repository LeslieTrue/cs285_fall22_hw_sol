from cs285.policies.MLP_policy import MLPPolicy
import torch
import numpy as np
from cs285.infrastructure import sac_utils
from cs285.infrastructure import pytorch_util as ptu
from torch import nn
from torch import optim
import itertools

class MLPPolicySAC(MLPPolicy):
    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=3e-4,
                 training=True,
                 log_std_bounds=[-20,2],
                 action_range=[-1,1],
                 init_temperature=1.0,
                 **kwargs
                 ):
        super(MLPPolicySAC, self).__init__(ac_dim, ob_dim, n_layers, size, discrete, learning_rate, training, **kwargs)
        self.log_std_bounds = log_std_bounds
        self.action_range = action_range
        self.init_temperature = init_temperature
        self.learning_rate = learning_rate

        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(ptu.device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.learning_rate)

        self.target_entropy = -ac_dim

    @property
    def alpha(self):
        # TODO: Formulate entropy term
        entropy = torch.exp(self.log_alpha) # alpha
        return entropy

    def get_action(self, obs: np.ndarray, sample=True) -> np.ndarray:
        # TODO: return sample from distribution if sampling
        # if not sampling return the mean of the distribution 
        if len(obs.shape) == 1:
            obs = obs[None]
        with torch.no_grad():
            observation = ptu.from_numpy(obs)
            action, action_mean, _ = self(observation)
            
            if sample:
                action_rt = action
            else:
                action_rt = action_mean
        
        action_rt = ptu.to_numpy(action_rt.clamp(*self.action_range))
        return action_rt

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: Implement pass through network, computing logprobs and apply correction for Tanh squashing

        # HINT: 
        # You will need to clip log values
        # You will need SquashedNormal from sac_utils file 
        # self.mean_net

        ##########################################

        ##TBD
        batch_size = observation.size()[0]
        logstd = self.logstd
        clipped = torch.clip(logstd, min = self.log_std_bounds[0], max = self.log_std_bounds[1]) # clipping log
        scale = torch.exp(clipped).repeat(batch_size, 1)# init squash function input

        action_distribution = sac_utils.SquashedNormal(loc = self.mean_net(observation), scale = scale)
        action_sample = action_distribution.rsample()

        # action_min, action_max = self.action_range
        # width = .5 * (action_max - action_min)
        # index = .5 * (action_max + action_min)
        # decompose action
        # action = action_sample * width + index
        action = action_sample
        # decompose mean action
        action_mean = action_distribution.mean
        # action_mean = action_mean * width + index
        
        log_prob = action_distribution.log_prob(action_sample)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, action_mean, log_prob

    def update_actor(self, ob_no, ac_na, next_ob_no, re_n, terminal_n, critic):
        ob_no = ptu.from_numpy(ob_no)
        # ac_na = ptu.from_numpy(ac_na)
        # next_ob_no = ptu.from_numpy(next_ob_no)
        # re_n = ptu.from_numpy(re_n).unsqueeze(1)
        # terminal_n = ptu.from_numpy(terminal_n).unsqueeze(1)  
        action, _, log_prob = self(ob_no)

        Q1, Q2 = critic(ob_no, action)
        Q = torch.minimum(Q1, Q2)
        actor_loss = (self.alpha.detach() * log_prob - Q).mean()
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()

        # calculate alpha loss
        alpha_loss = ( - self.alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return actor_loss.item(), alpha_loss.item(), self.alpha