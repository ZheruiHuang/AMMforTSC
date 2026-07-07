import math
import json
import os
import tempfile
import time
import atexit
from copy import deepcopy
from pathlib import Path

import numpy as np
import cityflow

from utils.read_roadnet import read_roadnet

_global_mode = 1

class World:
    def __init__(self, config_path, thread_num, max_time):
        config_file, engine_config_path = self._normalize_config(config_path)
        self.eng = cityflow.Engine(engine_config_path, thread_num)
        
        roadnet_path = os.path.join(config_file["dir"], config_file["roadnetFile"])
        self.roadnet_dict = read_roadnet(roadnet_path)
        
        self.max_time = max_time
        self.cur_time = 0
        self.inter_list = []  # get fix order intersection
        self.reward_list = []  # idx is the same as self.inter_list
        self.state_dim = 60
        self.phase_pass_veh_num = 20
        self.last_infos = []
        
        self.infos = ('grid_veh_num', 'grid_veh_speed', 'mid_cnt', 'end_cnt')
        self.aux_var = ('accum_mid_cnt', 'accum_end_cnt')
        self.aux_set = ('mid_set', 'end_set')
        
        self.set_inter_info()

    def _normalize_config(self, config_path):
        config_path = Path(config_path).expanduser().resolve()
        with open(config_path, 'r') as f:
            config_file = json.load(f)

        cityflow_root = config_path.parent.parent
        configured_dir = Path(config_file.get("dir", ""))
        if configured_dir.is_absolute():
            data_dir = configured_dir
        else:
            data_dir = (Path.cwd() / configured_dir).resolve()

        roadnet_path = data_dir / config_file["roadnetFile"]
        if not roadnet_path.exists():
            data_dir = cityflow_root

        config_file["dir"] = str(data_dir) + os.sep
        tmp_config = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="amm_cityflow_",
            delete=False,
        )
        with tmp_config:
            json.dump(config_file, tmp_config, indent=4)
        atexit.register(lambda path=tmp_config.name: os.path.exists(path) and os.remove(path))
        return config_file, tmp_config.name
        
    def set_inter_info(self):
        # set self.inter_list and inter_infos
        self.inter_list = [inter_id for inter_id in self.roadnet_dict['inter'] if not self.roadnet_dict['inter'][inter_id].virtual]
        self.reward_list = [float('-inf') for _ in range(len(self.inter_list))]
        self.last_infos = [{'accum_mid_cnt':np.zeros((8,1)), 'accum_end_cnt':np.zeros((8,1))} for _ in range(len(self.inter_list))]
        
        for inter_id in self.inter_list:
            self.roadnet_dict['inter'][inter_id].infos['grid_veh_num'] = np.zeros((8, self.state_dim))
            self.roadnet_dict['inter'][inter_id].infos['grid_veh_speed'] = np.zeros((8, self.state_dim))
            self.roadnet_dict['inter'][inter_id].infos['mid_cnt'] = np.zeros((8, 1))
            self.roadnet_dict['inter'][inter_id].infos['end_cnt'] = np.zeros((8, 1))
            self.roadnet_dict['inter'][inter_id].infos['aux_var']['accum_mid_cnt'] = np.zeros((8, 1))
            self.roadnet_dict['inter'][inter_id].infos['aux_var']['accum_end_cnt'] = np.zeros((8, 1))
            self.roadnet_dict['inter'][inter_id].infos['aux_set']['mid_set'] = [set() for _ in range(8)]
            self.roadnet_dict['inter'][inter_id].infos['aux_set']['end_set'] = [set() for _ in range(8)]
            
    def next_step(self, tl_phase_list=[]):
        # Input phases 0-7 map to CityFlow light phase indices 1-8.
        if self.cur_time >= self.max_time:
            raise RuntimeError('Current step is greater than maximum step!')
        assert len(self.inter_list) == len(tl_phase_list)
        for i, tl_phase in enumerate(tl_phase_list):
            self.eng.set_tl_phase(self.inter_list[i], 
                                  self.roadnet_dict['inter'][self.inter_list[i]].lightPhase_idx_mapping[tl_phase+1])
                
        self.eng.next_step()
        self.cur_time += 1
        self.update_inter_info()
        self.update_rewards()
        
    def reset(self):
        self.eng.reset()
        self.cur_time = 0
        self.reward_list = [float('-inf') for _ in range(len(self.inter_list))]
        self.last_infos = [{'accum_mid_cnt':np.zeros((8,1)), 'accum_end_cnt':np.zeros((8,1))} for _ in range(len(self.inter_list))]
        for inter_id in self.inter_list:
            for info in self.infos:
                shape = self.roadnet_dict['inter'][inter_id].infos[info].shape
                self.roadnet_dict['inter'][inter_id].infos[info] = np.zeros(shape)
            for var in self.aux_var:
                shape = self.roadnet_dict['inter'][inter_id].infos['aux_var'][var].shape
                self.roadnet_dict['inter'][inter_id].infos['aux_var'][var] = np.zeros(shape)
            for set_list in self.aux_set:
                for set in self.roadnet_dict['inter'][inter_id].infos['aux_set'][set_list]:
                    set.clear()
        
    def update_inter_info(self):
        self.update_grid()
        self.update_cnt()
        
    def update_grid(self):
        lane_veh_dict = self.eng.get_lane_vehicles()
        veh_dist_dict = self.eng.get_vehicle_distance()
        veh_speed_dict = self.eng.get_vehicle_speed()
        dir_list = ['north', 'east', 'south', 'west']
        for inter_id in self.inter_list:
            cur_grid_veh_num = np.zeros((8, self.state_dim))
            cur_grid_veh_speed = np.zeros((8, self.state_dim))
            row = 0
            for dir in dir_list:
                length = self.roadnet_dict['road'][self.roadnet_dict['inter'][inter_id].dir_inRoads[dir]].length
                for lane in [0, 1]:
                    for veh_id in lane_veh_dict[f'{self.roadnet_dict["inter"][inter_id].dir_inRoads[dir]}_{lane}']:
                        dist = veh_dist_dict[veh_id]
                        speed = veh_speed_dict[veh_id]
                        col = math.floor((length - dist) / 7.5)
                        if col >= self.state_dim:
                            continue
                        else:
                            cur_grid_veh_num[row, col] += 1
                            cur_grid_veh_speed[row, col] += speed
                            # note: avg_grid_veh_speed = grid_veh_speed / grid_veh_num
                    row += 1
            self.roadnet_dict['inter'][inter_id].infos['grid_veh_num'] = cur_grid_veh_num
            self.roadnet_dict['inter'][inter_id].infos['grid_veh_speed'] = cur_grid_veh_speed
    
    def update_cnt(self):
        # mid_grid = [math.ceil((self.state_dim-1)/2), math.ceil((self.state_dim-1)/2)-1]
        # end_grid = [self.state_dim-1, self.state_dim-2]
        lane_veh_dict = self.eng.get_lane_vehicles()
        veh_dist_dict = self.eng.get_vehicle_distance()
        dir_list = ['north', 'east', 'south', 'west']
        for inter_id in self.inter_list:
            row = 0
            for dir in dir_list:
                length = self.roadnet_dict['road'][self.roadnet_dict['inter'][inter_id].dir_inRoads[dir]].length
                max_grid = math.floor(length / 7.5)
                if max_grid >= self.state_dim:
                    mid_grid = [math.ceil((self.state_dim-1)/2), math.ceil((self.state_dim-1)/2)-1]
                    end_grid = [self.state_dim-1, self.state_dim-2]
                else:
                    mid_grid = [math.ceil((max_grid)/2), math.ceil((max_grid)/2)-1]
                    end_grid = [max_grid, max_grid-1]
                for lane in [0, 1]:
                    new_mid_set = set()
                    new_end_set = set()
                    for veh_id in lane_veh_dict[f'{self.roadnet_dict["inter"][inter_id].dir_inRoads[dir]}_{lane}']:
                        dist = veh_dist_dict[veh_id]
                        col = math.floor((length - dist) / 7.5)
                        for grid in mid_grid:
                            if col == grid:
                                if veh_id not in self.roadnet_dict['inter'][inter_id].infos['aux_set']['mid_set'][row]:
                                    self.roadnet_dict['inter'][inter_id].infos['aux_var']['accum_mid_cnt'][row, 0] += 1
                                new_mid_set.add(veh_id)
                        for grid in end_grid:
                            if col == grid:
                                if veh_id not in self.roadnet_dict['inter'][inter_id].infos['aux_set']['end_set'][row]:
                                    self.roadnet_dict['inter'][inter_id].infos['aux_var']['accum_end_cnt'][row, 0] += 1
                                new_end_set.add(veh_id)
                    self.roadnet_dict['inter'][inter_id].infos['aux_set']['mid_set'][row] = new_mid_set
                    self.roadnet_dict['inter'][inter_id].infos['aux_set']['end_set'][row] = new_end_set
                    row += 1
        
    # def update_rewards(self):
    #     lane_waiting_veh_cnt_dict = self.eng.get_lane_waiting_vehicle_count()
    #     for i, inter_id in enumerate(self.inter_list):
    #         sum_waiting_veh: int = 0
    #         lane_num = 0
    #         for road in self.roadnet_dict['inter'][inter_id].in_roads:
    #             for lane in [0, 1]:
    #                 sum_waiting_veh += lane_waiting_veh_cnt_dict[f'{road}_{lane}']
    #                 lane_num += 1
    #         reward = (-1) * sum_waiting_veh
    #         self.reward_list[i] = reward
    
    def update_rewards(self):
        avg_tt = self.eng.get_average_travel_time()
        for i, inter_id in enumerate(self.inter_list):
            self.reward_list[i] = avg_tt
    
    def get_ob(self, inter_idx):
        global _global_mode
        return self.infos_to_ob(inter_idx, _global_mode)
    
    def get_state(self, inter_idx):
        return self.roadnet_dict['inter'][self.inter_list[inter_idx]].infos['grid_veh_num']
    
    def get_infos(self, inter_idx):
        return self.roadnet_dict['inter'][self.inter_list[inter_idx]].infos
    
    def get_reward(self, inter_idx):
        return self.reward_list[inter_idx]
    
    def is_done(self, inter_idx):
        return (self.cur_time >= self.max_time)
    
    def infos_to_ob(self, inter_idx, mode):
        '''
        Returns:
            Observation.
            Shape: (8, x). 8: The number of lanes. x: The number of features.
        '''
        infos = self.get_infos(inter_idx)
        if mode == 1:
            ob = np.zeros((8, 3))
            grid_veh_num = infos['grid_veh_num']
            grid_veh_speed = infos['grid_veh_speed']
            
            for i in range(8):
                ob[i, 0] = np.sum(grid_veh_num[i])
                for j in range(1, 3):
                    ob[i, j] = np.mean(grid_veh_speed[i][j*self.phase_pass_veh_num:(j+1)*self.phase_pass_veh_num])
            
            return ob
        
        elif mode == 2:
            ob = np.sum(infos['grid_veh_num'], axis=1).reshape(8, 1)
            ob = np.concatenate((ob, infos['mid_cnt']), axis=1)
            assert ob.shape == (8, 2)
            return ob
        
        elif mode == 3:
            ob = np.sum(infos['grid_veh_num'], axis=1).reshape(8, 1)
            ob = np.concatenate((ob, infos['end_cnt']), axis=1)
            assert ob.shape == (8, 2)
            return ob
        
        else:
            raise ValueError(f'Function <infos_to_ob> gets an invalid value {mode} for argument <mode>.')
        
    def stat_grid_cnt(self):
        '''
        Count the number of vehicles that had passed middle grid and 
        ending grid between the last two calls to this function.
        '''
        for i, inter_id in enumerate(self.inter_list):
            self.roadnet_dict['inter'][inter_id].infos['mid_cnt'] = self.roadnet_dict['inter'][inter_id].infos['aux_var']['accum_mid_cnt'] - self.last_infos[i]['accum_mid_cnt']
            self.roadnet_dict['inter'][inter_id].infos['end_cnt'] = self.roadnet_dict['inter'][inter_id].infos['aux_var']['accum_end_cnt'] - self.last_infos[i]['accum_end_cnt']
            self.last_infos[i]['accum_mid_cnt'] = deepcopy(self.roadnet_dict['inter'][inter_id].infos['aux_var']['accum_mid_cnt'])
            self.last_infos[i]['accum_end_cnt'] = deepcopy(self.roadnet_dict['inter'][inter_id].infos['aux_var']['accum_end_cnt'])
            
        
