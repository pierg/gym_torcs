import gym
from gym import spaces
import numpy as np
# from os import path
import snakeoil3_gym as snakeoil3
import numpy as np
import copy
import collections as col
import os
import time
import math


class TorcsEnv:
    terminal_judge_start = 100  # Speed limit is applied after this step
    termination_limit_progress = 1  # [km/h], episode terminates if car is running slower than this limit
    default_speed = 300

    initial_reset = True


    def __init__(self, vision=False, throttle=False, gear_change=False):

        self.vision = vision
        self.throttle = throttle
        self.gear_change = gear_change

        self.initial_run = True

        # start torcs! (missleadning name!)
        self.reset_torcs()


    def step(self, a_t, early_stop):

        ## Create a desiered action with Effectors listed in same order as : arxiv.org/pdf/1304.1672.pdf table 3
        torcs_action = self.client.R.effectors # this is pass by reference so updating it updates the one in snakeoil

        # 1. accel in [0, 1]
        torcs_action.update({'accel': a_t[1]})

        # 2. brake in [0, 1]
        torcs_action.update({'brake': a_t[2]})

        # 3 clutch in [0, 1]
        # not implemented  ????
        torcs_action.update({'clutch': 0})

        # 4. gear in -1, 0, 1, ... , 6
        if(self.gear_change):
            gear = self.get_gear() #TODO this is just since the real automatic doesnt seem to work!!!!
        else:
            gear = 1
        torcs_action.update({'gear': gear})

        # 5. steer in [-1, 1]
        torcs_action.update({'steer': a_t[0]})

        # 6. focus in [-90,90] (list of 5 values) 1. should only be set once, 2. not reliable as sensors!
        #torcs_action.update({'focus': [-90,-45,0,45,90]})
        #torcs_action.update({'focus': [-50, -25, 0, 25, 50]}) # testing more forward vision!

        # 7. meta in 0,1 (restart race or not)
        torcs_action.update({'meta': 0})



        # Save the privious full-observation from torcs for the reward calculation
        prev_observation = copy.deepcopy(self.client.S.sensors)

        # Apply the Agent's action into torcs
        self.client.respond_to_server()

        # Get the response of TORCS
        self.client.get_servers_input()

        # Get the current full-observation from torcs
        # containing all 19 sensors from arxiv.org/pdf/1304.1672.pdf table 1/2 as key : value
        self.observation = self.client.S.sensors

        # calculate reward
        reward = self.calculate_reward(self.observation, prev_observation, early_stop)

        # scaled obs to send to agent
        obs = self.scale_observation(copy.deepcopy(self.client.S.sensors))

        done = (self.client.R.effectors['meta'] == 1)
        return obs, reward, done, {}

    def calculate_reward(self, obs, obs_pre, early_stop):

        # idea, everythin that is positive, should increse progress!
        # everything that is negative should decrease penalty!
        # reward = progress - penalty!!!

        penalty = 0 # for safety critic

        # Reward setting Here #######################################
        # direction-dependent positive reward
        track = np.array(obs['track'])
        trackPos = np.array(obs['trackPos'])
        sp = np.array(obs['speedX'])
        #damage = np.array(obs['damage'])
        #rpm = np.array(obs['rpm'])

        dist = obs['distFromStart'] - obs_pre['distFromStart']
        progress_old = sp * np.cos(obs['angle'])  - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos']) # OLD
        progress = sp * np.cos(obs['angle']) + dist

        ### penalty =  -(damage) -( "Y" speed) - ( dist from center)
        penalty = -(obs['damage'] - obs_pre['damage']) - np.abs(sp*np.sin(obs['angle'])) - sp * np.abs(obs['trackPos'])


        reward_old = progress_old

        # collision detection OLD
        if obs['damage'] - obs_pre['damage'] > 0:
            reward_old = -(obs['damage'] - obs_pre['damage'])


        # Termination judgement
        if (abs(track.any()) > 1 or abs(trackPos) > 1 and early_stop):  # Episode is terminated if the car is out of track
            reward_old = -200
            penalty -= 200
            print("META = 1 ... out of track")
            self.client.R.effectors['meta'] = 1

        if self.terminal_judge_start < self.time_step:  # Episode terminates if the progress of agent is small
            if((progress < self.termination_limit_progress) and early_stop ):
                #reward = -50
                penalty -= 50
                print("META = 1 ... Minimal Progress!")
                self.client.R.effectors['meta'] = 1

        if np.cos(obs['angle']) < 0:  # Episode is terminated if the agent runs backward
            print("META = 1 ... Running backwards")
            self.client.R.effectors['meta'] = 1

        if self.client.R.effectors['meta'] is 1:  # Send a reset signal
            self.initial_run = False
            self.client.respond_to_server()

        reward = progress + penalty
        self.time_step += 1
        return [reward, progress, penalty, reward_old]

    def reset(self, relaunch=False):
        #print("Reset")

        self.time_step = 0

        if self.initial_reset is not True:
            self.client.R.effectors['meta'] = True
            self.client.respond_to_server()

            ## TENTATIVE. Restarting TORCS every episode suffers the memory leak bug!
            if relaunch is True:
                self.reset_torcs()
                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs
        self.client.MAX_STEPS = np.inf

        client = self.client
        client.get_servers_input()  # Get the initial input from torcs

        self.observation = self.scale_observation(client.S.sensors)  # Get the current full-observation from torcs

        self.last_u = None

        self.initial_reset = False
        return self.observation

    def end(self):
        os.system('pkill torcs')

    def reset_torcs(self):
       #print("relaunch torcs")
        os.system('pkill torcs')
        #os.system('sysctl-w vm.drop_caches=3') line i use on my jetson tx1
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nolaptime -vision &')
        else:
            os.system('torcs -nofuel -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)


    def obs_vision_to_image_rgb(self, obs_image_vec):
        image_vec =  obs_image_vec
        r = image_vec[0:len(image_vec):3]
        g = image_vec[1:len(image_vec):3]
        b = image_vec[2:len(image_vec):3]

        sz = (64, 64)
        r = np.array(r).reshape(sz)
        g = np.array(g).reshape(sz)
        b = np.array(b).reshape(sz)
        return np.array([r, g, b], dtype=np.uint8)

    def scale_observation(self, raw_obs):
        ## listed all for future work but only scale the ones i need!!!
        raw_obs['angle'] = raw_obs['angle'] / math.pi
        # raw_obs['curLapTime']
        #raw_obs['damage'}]
        #raw_obs['distFromStart']
        #raw_obs['distRaced']
        raw_obs['focus'] = np.array(raw_obs['focus'])/200
        #raw_obs['fuel']
        #raw_obs['gear']
        #raw_obs['lastLapTime']
        raw_obs['opponents'] = np.array(raw_obs['opponents'])/200
        #raw_obs['racePos']
        raw_obs['rpm'] = raw_obs['rpm']/10000
        raw_obs['speedX'] = raw_obs['speedX'] / self.default_speed
        raw_obs['speedY'] = raw_obs['speedY'] / self.default_speed
        raw_obs['speedZ'] = raw_obs['speedZ'] / self.default_speed
        raw_obs['track'] = np.array(raw_obs['track']) / 200
        #raw_obs['trackPos'] = raw_obs['trackPos'] # should not be input to actor/critic, only in reward?!?!
        raw_obs['wheelSpinVel'] = np.array(raw_obs['wheelSpinVel']) / 100; #TODO Why? was like this in prev version?
        #raw_obs['z']

        if(self.vision):
            raw_obs['img'] = self.obs_vision_to_image_rgb(raw_obs['img'])

        return raw_obs

    def get_gear(self):
        speedX = self.client.S.sensors['speedX']
        gear = 1
        if speedX > 50:
            gear = 2
        if speedX > 80:
            gear = 3
        if speedX > 110:
            gear = 4
        if speedX > 140:
            gear = 5
        if speedX > 170:
            gear = 6
        return gear