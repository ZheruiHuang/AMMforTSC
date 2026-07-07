import json
import math
from itertools import permutations


class Intersection:
    def __init__(self, id):
        self.id = id
        self.coord = {'x':None, 'y':None}
        self.in_roads = []
        self.in_roads_angle = []
        self.dir_inRoads = {'north':None, 'east':None, 'south':None, 'west':None}
        self.inRoads_dir = {}
        self.light_phases = []
        self.road_links = []
        self.roadLinkIdx_movement_mapping = [] # idx: index of road_link; value: index of movement(0~11)
        self.lightPhase_idx_mapping = [None, None, None, None, None, None, None, None, None] # len(self.lightPhase_idx_mapping) == 9;
        self.infos = {'aux_var':{}, 'aux_set':{}}
        self.virtual = None
    
    
class Road:
    def __init__(self, id):
        self.id = id
        self.start_inter: int = -1
        self.end_inter: int = -1
        self.length: float = -1

    
class RoadnetReader:
    def __init__(self, roadnet_f):
        with open(roadnet_f, 'r') as f:
            self.roadnet_dict = json.load(f)
        self.inter_dict = {}
        self.road_dict = {}
        
    def _read_intersections_and_roads(self):
        # read intersections
        # get intersections' id, coord, light_phases, road_links, virtual
        for inter in self.roadnet_dict['intersections']:
            new_inter = Intersection(inter['id'])
            new_inter.coord['x'], new_inter.coord['y'] = inter['point']['x'], inter['point']['y']
            new_inter.light_phases = inter['trafficLight']['lightphases'] # list
            new_inter.road_links = inter['roadLinks'] # list
            new_inter.virtual = inter['virtual']
            
            self.inter_dict[new_inter.id] = new_inter
        
        # read roads
        # get roads' id, start_inter, end_inter, length and intersections' in_roads, in_lanes
        for road in self.roadnet_dict['roads']:
            new_road = Road(road['id'])
            new_road.start_inter = road['startIntersection']
            new_road.end_inter = road['endIntersection']
            
            self.inter_dict[new_road.end_inter].in_roads.append(new_road.id)
            
            # length
            new_road.length = 0.0
            point_0 = None
            for point in road['points']:
                if point_0 == None:
                    point_0 = point
                else:
                    new_road.length += math.sqrt(math.pow(point['x']-point_0['x'], 2) + math.pow(point['y']-point_0['y'], 2))
                    point_0 = point
            
            self.road_dict[new_road.id] = new_road
    
    def _calc_inRoadsAngle(self):
        # calculate in_roads_angle
        for inter_id in self.inter_dict:
            to_x, to_y = self.inter_dict[inter_id].coord['x'], self.inter_dict[inter_id].coord['y']
            for in_road_id in self.inter_dict[inter_id].in_roads:
                from_x, from_y = self.inter_dict[self.road_dict[in_road_id].start_inter].coord['x'], self.inter_dict[self.road_dict[in_road_id].start_inter].coord['y']
                if from_x == to_x:
                    if from_y > to_y:
                        self.inter_dict[inter_id].in_roads_angle.append(0.5*math.pi)
                    else: # from_y < to_y
                        self.inter_dict[inter_id].in_roads_angle.append(1.5*math.pi)
                else:   
                    alpha = math.atan((from_y - to_y) / (from_x - to_x))
                    if alpha > 0 and from_y < to_y:
                        alpha += math.pi
                    elif alpha < 0 and from_y > to_y:
                        alpha += math.pi
                    elif alpha < 0 and from_y < to_y:
                        alpha += 2*math.pi
                    self.inter_dict[inter_id].in_roads_angle.append(alpha)
                    
            assert len(self.inter_dict[inter_id].in_roads) == len(self.inter_dict[inter_id].in_roads_angle)
        
    def _calc_inRoadsDir(self):
        # calculate direction of in_roads
        # get intersections' inRoads_dir, dir_inRoads
        idx_list = [0, 1, 2, 3]
        dir_list = ['north', 'east', 'south', 'west']
        for inter_id in self.inter_dict:
            assert len(self.inter_dict[inter_id].in_roads) <= 4
            min_loss = 4*math.pi
            best_idx_order = tuple()
            for idx_order in permutations(idx_list, 4):
                loss = self._calc_angle_loss(inter_id, idx_order, dir_list)
                if loss <= min_loss:
                    best_idx_order = idx_order
                    min_loss = loss
            for i in range(len(self.inter_dict[inter_id].in_roads)):
                self.inter_dict[inter_id].dir_inRoads[dir_list[best_idx_order[i]]] = self.inter_dict[inter_id].in_roads[i]
                # inRoads_dir
                self.inter_dict[inter_id].inRoads_dir[self.inter_dict[inter_id].in_roads[i]] = dir_list[best_idx_order[i]]
                
    def _calc_angle_loss(self, inter_id, idx_order, dir_list):
        dir_angle_dict = {'north':0.5*math.pi, 'east':0.0, 'south':1.5*math.pi, 'west':math.pi}
        loss = 0.0
        for i, in_road_angle in enumerate(self.inter_dict[inter_id].in_roads_angle):
            one_dir_loss = abs(dir_angle_dict[dir_list[idx_order[i]]] - in_road_angle)
            if one_dir_loss > math.pi:
                one_dir_loss = 2*math.pi - one_dir_loss
            loss += one_dir_loss
        return loss
    
    def _get_lightPhaseIdxMapping(self):
        # get intersections' lightPhase_idx_mapping
        self._get_roadLinkIdx_movement_mapping()
        for inter_id in self.inter_dict:
            for i, phase in enumerate(self.inter_dict[inter_id].light_phases):
                s = set()
                for link_idx in phase['availableRoadLinks']:
                    s.add(self.inter_dict[inter_id].roadLinkIdx_movement_mapping[link_idx])
                light_phase = self._get_light_phase_idx(s)
                self.inter_dict[inter_id].lightPhase_idx_mapping[light_phase] = i
        
    def _get_roadLinkIdx_movement_mapping(self):
        # get intersections' roadLinkIdx_movement_mapping
        for inter_id in self.inter_dict:
            for road_link in self.inter_dict[inter_id].road_links:
                self.inter_dict[inter_id].roadLinkIdx_movement_mapping.append(self._get_movementIdx(self.inter_dict[inter_id].inRoads_dir[road_link['startRoad']], road_link['type']))
    
    def _get_movementIdx(self, inRoad_dir, link_type):
        dir_dict = {'north':0, 'east':1, 'south':2, 'west':3}
        type_dict = {'turn_right':0, 'go_straight':1, 'turn_left':2}
        return dir_dict[inRoad_dir]*3 + type_dict[link_type]
    
    def _get_light_phase_idx(self, set): # set is available_roadLinks_corresponding_movement_set
        phase_dict = {1:[2,8], 2:[1,7], 3:[5,11], 4:[4,10], 5:[1,2], 6:[4,5], 7:[7,8], 8:[10,11]} # 0:[]
        if len(set) == 0:
            return 0
        else:
            for phase_idx in phase_dict:
                flag = True
                for movement in phase_dict[phase_idx]:
                    if movement not in set:
                        flag = False
                if flag == True:
                    return phase_idx
        return 0
    
    def read_roadnet(self):
        self._read_intersections_and_roads()
        self._calc_inRoadsAngle()
        self._calc_inRoadsDir()
        self._get_lightPhaseIdxMapping()
        return {'inter':self.inter_dict, 'road':self.road_dict}

def read_roadnet(roadnet_f):
    r = RoadnetReader(roadnet_f)
    return r.read_roadnet()
