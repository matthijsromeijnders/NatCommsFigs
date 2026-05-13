
import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize
import pandas as pd
from tqdm import tqdm
import warnings
import networkx as nx
import pickle
from delaybuffernetwork import DelayBufferNetwork
import bisect
from Service import ServiceSheet
warnings.filterwarnings("ignore")

class DBN_static_delay(DelayBufferNetwork):
    def __init__(self, from_df = None, random_event_dict=False, load=False, path="", system="dict",
                 dont_build_df=False, del_df=False, uniform_time_range=False, block_lens=None,
                 recovery_rate=0.00):
        super().__init__(from_df = from_df, random_event_dict=random_event_dict, load=load, path=path, system=system,
                 dont_build_df=dont_build_df, del_df=del_df, uniform_time_range=uniform_time_range)
           
        self.build_event_dict2()
        self.service_sheet = ServiceSheet.from_df(from_df, event_dict = self.event_dict)   
        self.add_next_event_id()
        self.make_last_event_arr()
        self.accumulated_delay = np.zeros((len(self.unique_agents)))
        self.unique_times = np.unique(self.network["t"].to_numpy()) 
        self.effective_agent_delays = np.zeros((self.total_events, len(self.unique_agents)))
        self.delay_list = []
        self.avg_delay_of_active_agents = np.array([])
        self.delay_minus_buffer_per_event = np.zeros(len(self.event_dict))
        self.final_events = self.network.groupby("i").last()
        self.block_ids = np.unique(self.network["block"]).astype(int)
        self.agent_delays = np.zeros((self.total_events+1, len(self.unique_agents)))
        self.block_lens = block_lens
        if self.block_lens is None:
            self.block_lens = np.ones(len(self.block_ids))
        self.recovery_rate = recovery_rate
          


    def build_event_dict2(self):
        """Build event dictionary from a df, and event time array. Necessary for running process_delays_dict().
        """
        self.event_dict = {}
        self.time_event_dict = {}
        self.total_events = len(np.unique(self.network.event_id+1))
        self.event_time_array = np.zeros((self.total_events,2))
        self.unique_agents = np.unique(np.concatenate((self.network["i"], self.network["j"])))

        for _, row in self.network.iterrows():
            i = row['i']
            j = row['j']
            t = row['t']
            buffer = row["buffer"]
            delay = row["delay"]
            event_id = row['event_id']
            block_id = row['block']
            if event_id not in self.event_dict:
                if i == j:
                    raise ValueError("i = j")
                # If not, add it to the dictionary and create a new set to hold unique (i,j,t) values
                self.event_dict[event_id] = {"agents": [int(i), int(j)], "t": float(t), 
                                             "random_delay": float(delay), 
                                             "buffer": float(buffer), "block": int(block_id),
                                            "effective_delay": 0, "accumulated_delay_of_train": 0}
                self.event_time_array[int(event_id),:] = np.array([int(event_id), t])
        self.time_event_dict = {}
        for _, row in self.network.iterrows():
            i = row['i']
            j = row['j']
            t = row['t']
            buffer = row["buffer"]
            delay = row["delay"]
            event_id = row['event_id']

            self.time_event_dict.setdefault(str(t), []).append(event_id)
    def add_next_event_id(self):
        """Adds the next event id to the event dictionary. This is used to propagate delays and buffers.
        """
        for event_id in range(self.total_events):
            curr_event_dict = self.event_dict[event_id]
            agent = curr_event_dict["agents"][0]
            next_event_id = self.service_sheet.get_next_event_id(agent, event_id)
            self.event_dict[event_id]["next_event_id"] = next_event_id
            secondary_agent = curr_event_dict["agents"][1]
            next_event_id_secondary = self.service_sheet.get_next_event_id(secondary_agent, event_id)
            self.event_dict[event_id]["next_event_id_secondary"] = next_event_id_secondary
    

    def make_last_event_arr(self):
        self.last_event_arr = np.zeros(len(self.unique_agents))
        for agent in self.unique_agents:
            last_event = int(self.service_sheet.get_last_event_id(agent))
            self.last_event_arr[agent] = last_event



    def set_buffer(self, events, buffers):
        for i in range(len(events)):
            event_id = events[i]
            buffer = buffers[i]
            self.event_dict[event_id]["buffer"] = buffer


    def add_delay(self, tau=1, add=0):
        """Adds a delay to a delay column in the temporal network Pandas.DataFrame format. Multiple options are possible to add delays.

        Args:
            tau (float) : lambda variable of exponential distribution.
            using_dict(bool, optional): Set true to add the delays to the dict-based graph system. Defaults to True.
            using_df(bool, optional): Set true to add the delays to the array-based system. Defaults to False.
            using_df(bool, optional): Set true to add the delays to the DF system. Defaults to False.
        """
        # Add delays to dict-based graph system.
        # self.agent_delays[0,:] = scipy.stats.expon.rvs(scale=tau, loc=0, size = len(self.agent_delays[0,:]))
        self.tau = tau
        if self.dict:
            delay = scipy.stats.expon.rvs(scale=tau, loc=0, size = self.total_events) + add
            self.mean_delay = tau
            delay = np.where(delay < 0, 0, delay)
            for i in range(self.total_events):

                self.event_dict[i]["random_delay"] = delay[i]
                self.delay_list.append(delay[i])


    def reset(self):
        self.agent_delays *= 0
        self.accumulated_delay *= 0

    def process_delays_dict(self, buffer_addition=0, posbuffer=True):
        """
        Processes / propagates delays and buffers throughout the network, uses the event dictionaries for faster processing.
        """
        self.total_delay_in_system = np.zeros((self.total_events))
        self.mean_delay_in_system = np.zeros((self.total_events))
        self.event_delay_in_system = np.zeros((self.total_events))
        self.active_agents_delay = np.zeros((self.total_events))
        self.active_agents_delay_mean = np.zeros((self.total_events))
        started_agents = np.zeros((len(self.unique_agents)))
        self.active_agents = np.zeros((self.total_events))
        self.agents_delayed = np.zeros((self.total_events, len(self.unique_agents)))
        for unique_event_iterator in range(self.total_events):
            current_event_dict = self.event_dict[unique_event_iterator]
            
            # Gather all unique agents.
            primary_agent = current_event_dict["agents"][0]
            secondary_agent = int(current_event_dict["agents"][1])
            block_id = current_event_dict["block"]
            block_len = self.block_lens[block_id]
            event_time = current_event_dict["t"]
            next_event_id = current_event_dict["next_event_id"]
            next_event_time = self.event_dict[next_event_id]["t"]
            next_event_id_secondary = current_event_dict["next_event_id_secondary"]

            scheduled_running_time = max(0,next_event_time - event_time)
            recovery_time = scheduled_running_time * self.recovery_rate
            if primary_agent == secondary_agent:
                raise ValueError("Primary agent is equal to secondary agent.")
            # Get the delays and buffers we need.
            
            # Try with previous acc delay.
            primary_accumulated_delay_start = self.accumulated_delay[int(primary_agent)] 
            primary_accumulated_delay = primary_accumulated_delay_start - recovery_time
            secondary_accumulated_delay = self.accumulated_delay[int(secondary_agent)] 
            
            random_delay = current_event_dict["random_delay"] * block_len
            buffer_in_event = current_event_dict["buffer"] + buffer_addition
            buffer_in_event = max(0, buffer_in_event)
            accumulated_delay_diff = primary_accumulated_delay - secondary_accumulated_delay
            delay_minus_buffer = accumulated_delay_diff - buffer_in_event#- buffer_addition

            # Positional buffer is the amount of time that the secondary agent is delayed more than the primary.
            
            if posbuffer:
                positional_buffer =  max(0, secondary_accumulated_delay - primary_accumulated_delay)
            else:
                positional_buffer = 0
            prim_delay_that_can_be_passed_on = max(0, delay_minus_buffer + random_delay) - positional_buffer
            
            # Can be pos or neg. Is kept track of correctly with the positional buffer. Need to debug leftover recovery time.
            propagated_delay_to_secondary = prim_delay_that_can_be_passed_on
            # Increase accumulated delay array
            self.accumulated_delay[int(primary_agent)] = max(0, self.accumulated_delay[int(primary_agent)] - recovery_time+ random_delay) 
            secondary_accdelay_after = max(0, secondary_accumulated_delay + propagated_delay_to_secondary)
            self.accumulated_delay[int(secondary_agent)] = secondary_accdelay_after
            prim = self.accumulated_delay[int(primary_agent)] 
            prop = secondary_accdelay_after - secondary_accumulated_delay

            self.agent_delays[unique_event_iterator:,int(secondary_agent)] = secondary_accdelay_after
            self.agent_delays[unique_event_iterator:,int(primary_agent)] = prim
            
            # new array to see only if primary agents are delayed wrt the schedule:
            last_event = int(self.last_event_arr[int(primary_agent)])
            try:
                self.agents_delayed[unique_event_iterator:last_event, int(primary_agent)] = max(0, primary_accumulated_delay + random_delay - buffer_in_event)
            except TypeError:
                print(last_event)
                raise TypeError

            # # Clamp negative delays to 0.
            sliced = self.agent_delays[unique_event_iterator:,int(secondary_agent)]
            self.agent_delays[unique_event_iterator:,int(secondary_agent)] = np.where(sliced < 0, 0, sliced)
            sliced = self.agent_delays[unique_event_iterator:,int(primary_agent)]
            self.agent_delays[unique_event_iterator:,int(primary_agent)] = np.where(sliced < 0, 0, sliced)
            
            self.event_dict[unique_event_iterator]["prop_delay"] = prop
            self.event_dict[unique_event_iterator]["prim_delay"] = max(0,random_delay-recovery_time)
            
            self.total_delay_in_system[unique_event_iterator] = np.sum(self.accumulated_delay)
            self.mean_delay_in_system[unique_event_iterator] = np.mean(self.accumulated_delay)
            
            # only count secondary if not first:
            if started_agents[int(primary_agent)] ==1 and started_agents[int(secondary_agent)] == 0:
                self.event_delay_in_system[unique_event_iterator] += np.sum(self.accumulated_delay[int(secondary_agent)]) + np.sum(self.accumulated_delay[int(primary_agent)])
                if int(next_event_id) == unique_event_iterator:
                    started_agents[int(primary_agent)] = 0
                if int(next_event_id_secondary) == unique_event_iterator:
                    started_agents[int(secondary_agent)] = 0
            elif started_agents[int(primary_agent)] == 0 and started_agents[int(secondary_agent)] == 1:
                self.event_delay_in_system[unique_event_iterator] += np.sum(self.accumulated_delay[int(primary_agent)])
                started_agents[int(secondary_agent)] = 1
                if int(next_event_id) == unique_event_iterator:
                    started_agents[int(primary_agent)] = 0
                if int(next_event_id_secondary) == unique_event_iterator:
                    started_agents[int(secondary_agent)] = 0
            else:
                started_agents[int(primary_agent)] = 1
                started_agents[int(secondary_agent)] = 1
                if int(next_event_id) == unique_event_iterator:
                    started_agents[int(primary_agent)] = 0
                if int(next_event_id_secondary) == unique_event_iterator:
                    started_agents[int(secondary_agent)] = 0
            self.active_agents[unique_event_iterator] = len(started_agents[started_agents == 1])
            
            self.active_agents_delay_mean[unique_event_iterator] = np.sum(self.accumulated_delay[started_agents == 1]) / len(started_agents[started_agents == 1])
            self.active_agents_delay[unique_event_iterator] = np.sum(self.accumulated_delay[started_agents == 1])
        # print("Done, now doing metrics")
        # Delays of active agents.
        self.mean_delay_of_active_agents = [] 
        self.mean_v_act = []
        for t, event_time in enumerate(self.unique_times):
            events_at_time = self.time_event_dict[str(event_time)] 
            agents_this_step = []
            for event in events_at_time:
                agent0 = int(self.event_dict[event]["agents"][0])
                agent1 = int(self.event_dict[event]["agents"][1])
                agents_this_step.append(agent0)
                agents_this_step.append(agent1)
            mean_delay_at_t = np.mean(self.agent_delays[t,agents_this_step])
            if t > 1:
                mean_V_act = np.mean(self.agent_delays[t,agents_this_step] - self.agent_delays[t-1,agents_this_step] )#+ self.agent_delays[0,agents_this_step])
                self.mean_v_act.append(mean_V_act)
            self.mean_delay_of_active_agents.append(mean_delay_at_t)
        # self.mean_v_act[-1] += np.mean(scipy.stats.expon.rvs(scale=self.tau, loc=0, size = len(self.unique_agents)))
        self.mean_delay_of_active_agents = np.array(self.mean_delay_of_active_agents)
        self.mean_v_act = np.array(self.mean_v_act)
                
        # blocking sections keep track 
        self.spatial_event_series = np.zeros((len(self.block_ids), self.total_events)) 
        self.spatial_event_series_prim = np.zeros((len(self.block_ids), self.total_events))
        self.total_prop_delay = 0
        self.total_prim_delay = 0
        started_agents = np.zeros((len(self.unique_agents)))
        for t, event_time in enumerate(self.unique_times):
            events_at_time = np.array(self.time_event_dict[str(event_time)]).astype(int)
            for event in events_at_time:
                agents_at_event = self.event_dict[event]["agents"]
                primary_agent = agents_at_event[0]
                secondary_agent = agents_at_event[1]
                block_id = self.event_dict[event]["block"]
                if block_id == -1:
                    continue
                else:
                    if started_agents[int(secondary_agent)] == 0: # we dont count propagation of the first event.
                        started_agents[int(secondary_agent)] = 1
                        started_agents[int(primary_agent)] = 1
                        self.spatial_event_series_prim[block_id, t] += self.event_dict[event]["prim_delay"] 
                        #self.spatial_event_series[block_id, t] += self.event_dict[event]["prop_delay"]
                        
                    else:
                        started_agents[int(primary_agent)] = 1 
                        self.spatial_event_series[block_id, t] += self.event_dict[event]["prop_delay"]        
                        self.spatial_event_series_prim[block_id, t] += self.event_dict[event]["prim_delay"] 
                        self.total_prop_delay += self.event_dict[event]["prop_delay"]
                        self.total_prim_delay += self.event_dict[event]["prim_delay"]
                        
                





                