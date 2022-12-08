import torch
import torch.nn.functional as F
import torch.optim as optim
from model import Actor, Critic
import torch.nn.utils.prune as prune

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class TD3:
    def __init__(self, lr, state_dim, action_dim, max_action):
        
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        print(self.actor)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        
        self.critic_1 = Critic(state_dim, action_dim).to(device)
        print(self.critic_1)
        self.critic_1_target = Critic(state_dim, action_dim).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=lr)
        
        self.critic_2 = Critic(state_dim, action_dim).to(device)
        self.critic_2_target = Critic(state_dim, action_dim).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=lr)
        
        self.max_action = max_action

        self.parameters_actor = (
                    (self.actor.l1, "weight"),
                    (self.actor.l2, "weight"),
                    (self.actor.l3, "weight"),
                    # (self.actor.l4, "weight"),
                    # (self.actor.l5, "weight"),
                    # (self.actor.l6, "weight"),
                    )
        self.parameters_actor_target = (
                    (self.actor_target.l1, "weight"),
                    (self.actor_target.l2, "weight"),
                    (self.actor_target.l3, "weight"),
                    # (self.actor_target.l4, "weight"),
                    # (self.actor_target.l5, "weight"),
                    # (self.actor_target.l6, "weight"),
                    )

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, replay_buffer, n_iter, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay):
        
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = replay_buffer.sample(batch_size)
            # print('here',next_state.shape)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((batch_size,1)).to(device)
            
            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * gamma * target_Q).detach()
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates:
            if i % policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (polyak * target_param.data) + ((1-polyak) * param.data))
        return actor_loss.cpu().data.numpy(), loss_Q1.cpu().data.numpy(), loss_Q2.cpu().data.numpy()
                
    def save(self, directory, name, ep):
        torch.save(self.actor.state_dict(), '%s/%s_actor_ep%s.pth' % (directory, name, ep))
        torch.save(self.actor_target.state_dict(), '%s/%s_actor_target_ep%s.pth' % (directory, name, ep))
        
        torch.save(self.critic_1.state_dict(), '%s/%s_crtic_1_ep%s.pth' % (directory, name, ep))
        torch.save(self.critic_1_target.state_dict(), '%s/%s_critic_1_target_ep%s.pth' % (directory, name, ep))
        
        torch.save(self.critic_2.state_dict(), '%s/%s_crtic_2_ep%s.pth' % (directory, name, ep))
        torch.save(self.critic_2_target.state_dict(), '%s/%s_critic_2_target_ep%s.pth' % (directory, name, ep))
        
    def load(self, directory, name, ep):
        self.actor.load_state_dict(torch.load('%s/%s_actor_ep%s.pth' % (directory, name, ep), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target_ep%s.pth' % (directory, name, ep), map_location=lambda storage, loc: storage))
        
        self.critic_1.load_state_dict(torch.load('%s/%s_crtic_1_ep%s.pth' % (directory, name, ep), map_location=lambda storage, loc: storage))
        self.critic_1_target.load_state_dict(torch.load('%s/%s_critic_1_target_ep%s.pth' % (directory, name, ep), map_location=lambda storage, loc: storage))
        
        self.critic_2.load_state_dict(torch.load('%s/%s_crtic_2_ep%s.pth' % (directory, name, ep), map_location=lambda storage, loc: storage))
        self.critic_2_target.load_state_dict(torch.load('%s/%s_critic_2_target_ep%s.pth' % (directory, name, ep), map_location=lambda storage, loc: storage))
        
        
    def load_actor(self, directory, name, ep):
        self.actor.load_state_dict(torch.load('%s/%s_actor_ep%s.pth' % (directory, name, ep), map_location=lambda storage, loc: storage))
        self.actor_target.load_state_dict(torch.load('%s/%s_actor_target_ep%s.pth' % (directory, name, ep), map_location=lambda storage, loc: storage))

    def prune_model(self):
        prune.global_unstructured(self.parameters_actor,pruning_method=prune.L1Unstructured,amount=0.1)
        prune.global_unstructured(self.parameters_actor_target,pruning_method=prune.L1Unstructured,amount=0.1)
