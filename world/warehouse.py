from __future__ import annotations
import os
from typing import Optional, List, TYPE_CHECKING

import pandas as pd
from lib.types.directed_graph import DirectedGraph
from world.layout import Layout
from world.landscape import Landscape
from lib.math import *
from world.managers.intersection_manager import IntersectionManager
from world.managers.order_manager import OrderManager
from world.managers.zone_manager import ZoneManager
from world.managers.job_manager import JobManager
from world.managers.robot_manager import RobotManager
from world.managers.pod_manager import PodManager
from world.managers.area_path_manager import AreaPathManager
from world.managers.station_manager import StationManager
from world.entities.order import Order
from world.entities.pod import Pod
from world.entities.robot import Robot
from world.entities.job import Job
from lib.generator.order_generator import *
from lib.constant import *
if TYPE_CHECKING:
    from world.entities.object import Object

class Warehouse:
    DIMENSION = 60
    def __init__(self):
        self._tick = 0
        self.ignored_types = ["pod", "station", "area_path", "intersection"]
        self.job_queue = []
        self.stop_and_go = 0
        self.total_energy = 0
        self.total_pod = 0
        self.total_turning = 0
        self.warehouse_size = []
        self.layout = Layout()
        self.landscape = Landscape(self.DIMENSION)
        self.order_manager = OrderManager(self)
        self.zone_manager = ZoneManager(self)
        self.job_manager = JobManager(self)
        self.intersection_manager = IntersectionManager(self, self.landscape.current_date_string)
        self.area_path_manager = AreaPathManager(self)
        self.robot_manager = RobotManager(self)
        self.pod_manager = PodManager(self)
        self.station_manager = StationManager(self)
        self.next_process_tick = 0
        self.update_intersection_using_RL = False
        self.zoning = False

        self.robot_using_RL = False
        self.rl_state = {}
        
        self.graph = DirectedGraph()
        self.graph_pod = DirectedGraph()
        self.updated_assigned_order = True
        
    def initWarehouse(self):
        self.robot_manager.initRobotManager()
        self.station_manager.initStationManager()
        self.pod_manager.initPodManager()
        # area path and intersection entity don't need connection back to the managers
        # self.area_path_manager.initAreaPathManager()
        # self.intersection_manager.initIntersectionManager()
        
        # Robot RL
        self.intersection_coords = self.intersection_manager.getAllIntersectionCoordinates()
        self.action_coords = [(x - 3, y) for x, y in self.intersection_coords] + [(x, y - 3) for x, y in self.intersection_coords if x > 0 and y > 0]
        self.action_coor_dict = {coord : idx for idx, coord in enumerate(self.action_coords)}

    def setWarehouseSize(self, size):
        self.warehouse_size = size

    def getWarehouseSize(self):
        return self.warehouse_size

    def getObjects(self):
        result = []
        result.extend(self.area_path_manager.getAllAreaPaths())
        result.extend(self.intersection_manager.getAllIntersections())
        result.extend(self.pod_manager.getAllPods())
        result.extend(self.robot_manager.getAllRobots())
        result.extend(self.station_manager.getAllStations())
        return result
    
    def getMovableObjects(self):
        result = []
        for o in self.getObjects():
            if o.object_type not in self.ignored_types or self._tick == 0:
                result.append(o)
        return result

    def tick(self):
        if int(self._tick) == self.next_process_tick:
            self.findNewOrders()
            self.processOrders() # hasilnya berupa  job queue
            if self.update_intersection_using_RL:
                self.intersection_manager.updateDirectionUsingDQN(int(self._tick))
        if len(self.job_queue) > 0:
            current_distance = 1000000
            nearest_robot: Optional[Robot] = None

            # Assign job to robot
            for o in self.getMovableObjects():
                if len(self.job_queue) > 0:
                    job: Job = self.job_queue[0]

                    if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                        dist = calculate_distance(o.pos_x, o.pos_y, job.pod_coordinate.x, job.pod_coordinate.y)
                        if dist < current_distance:
                            nearest_robot = o
                            current_distance = dist

            if nearest_robot is not None:
                job: Job = self.job_queue.pop(0)
                nearest_robot.assignJobAndSetToTakePod(job)
            # /Assign job to robot

        # Ngitung energy + replenishment
        total_energy = 0
        total_turning = 0
            
        robot_info  = []
        global_robot_positions = []
        robot_velocity = []
        robot_in_path = []
        avg_velocity_in_path = []
        robot_eta = []

        # Process robot menyelesaikan job dari assign robot 
        for o in self.getMovableObjects():
            initial_velocity = o.velocity
            o.move()
            if isinstance(o, Robot):
                total_energy += o.energy_consumption
                total_turning += o.turning
                if o.velocity == 0 and initial_velocity > 0:
                    self.stop_and_go += 1

                if o.job is not None and o.job.picking_delay == 0 and not o.job.is_finished:
                    need_replenish_pod = self.finishTaskInJob(o.job) # main function nyelesaiin job
                    if need_replenish_pod:
                        print(f"cihuy masuk")
                        pod: Pod = self.pod_manager.getPodsByCoordinate(o.job.pod_coordinate.x, o.job.pod_coordinate.y)
                        station_replenish = self.station_manager.findAvailableReplenishmentStation()
                        new_job = self.job_manager.createJob(pod.coordinate, station_id=station_replenish.id)
                        new_job.addReplenishmentTask(pod)
                        o.assignJobAndSetToStation(new_job)
                        

                if o.current_state == 'idle' and o.job is not None:
                    self.pod_manager.setPodAvailable(o.job.pod_coordinate)
                    o.job = None

                # Generate RL State - Robot Information
                global_robot_positions.append(f'{o.pos_x},{o.pos_y}')
                robot_velocity.append(o.velocity)
                robot_info.append([
                    # o.id,
                    o.heading / 360,
                    # o.shape,
                    o.velocity / 1.5,
                    (o.acceleration + 1 )/ 2,
                    o.pos_x / 60,
                    o.pos_y / 60,
                    o.dest.x / 60,
                    o.dest.y / 60,
                    o.color / 57,
                ])
            
        # Generate RL State - Path Information
        global_congestion_map = np.zeros((self.DIMENSION, self.DIMENSION))
        for o in self.getMovableObjects():
            path_velocity = 1.5
            if isinstance(o, Robot):
                if self.action_coor_dict.get((int(o.pos_x), int(o.pos_y))) is not None:
                    o.approaching_intersection = True
                num_of_robots = 0

                robot_positions_dict = {tuple(pos): idx for idx, pos in enumerate(global_robot_positions)}
                for coordinate_sequence, coordinate in enumerate(o.path):
                    robot_id = robot_positions_dict.get(tuple(coordinate))
                    if robot_id is not None:
                        path_velocity += (robot_velocity[robot_id] - 1.5) / (coordinate_sequence + 1)
                        num_of_robots += 1
                total_distance = len(o.path)
                estimated_time = total_distance / max(0.1, path_velocity)
                
                robot_in_path.append(num_of_robots / 30)
                avg_velocity_in_path.append(max(0.1, path_velocity) / 1.5)
                robot_eta.append(estimated_time * TICK_TO_SECOND / 60)

                for coordinate in o.path:
                    coordinate = [int(i) for i in coordinate.split(',')]
                    global_congestion_map[coordinate[0], coordinate[1]] += 1
        
        # Generate RL Action
        self.rl_state['robot_info'] = np.transpose(np.array(robot_info))
        self.rl_state['local_path_info'] = np.array([robot_in_path, avg_velocity_in_path, robot_eta])
        self.rl_state['global_congestion_map'] = global_congestion_map / 30 # Normalize
        
        for o in self.getMovableObjects():
            if isinstance(o, Robot):
                if o.approaching_intersection and self.robot_using_RL:
                    action = o.robot_manager.rl_agents[o.rl_model].act(self.rl_state)
                else:
                    action = 0

        self.total_energy = total_energy
        self.total_turning = total_turning
        # /Ngitung energy + replenishment

        if int(self._tick) == self.next_process_tick:
            self.next_process_tick += 1
            if self.update_intersection_using_RL:
                self.intersection_manager.updateModelAfterExecution(self._tick)

        self._tick += TICK_TO_SECOND

    def finishTaskInJob(self, job: Job):
        job_station = self.station_manager.getStationById(job.station_id)
        if job_station.isPickerStation():
            return self.finishPickingTask(job) # concrete implementation cara menyelesaikan task nya
        elif job_station.isReplenishmentStation():
            return self.finishReplenishmentTask(job)
    
    def finishPickingTask(self, job: Job):
        pod: Pod = self.pod_manager.getPodsByCoordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        sku_need_replenished = []
        for order_id, sku, quantity in job.orders:
            order: Order = self.order_manager.getOrderById(order_id)
            order.deliverQuantity(sku, quantity)
            print("order, sku, quantity :" ,order_id, sku, quantity)

            pod.pickSKU(sku, quantity)

            # Check for SKU Replenishment
            self.pod_manager.reduceSKUData(sku, quantity)
            sku, replenished_status = self.pod_manager.isSKUNeedReplenishment(sku)

            # SKU Replenished Triggered
            if(replenished_status == True): sku_need_replenished.append(sku)
    
            file_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
            assign_order_df = pd.read_csv(file_path)
            assign_order_df.loc[((assign_order_df['order_id'] == order.id) & (assign_order_df['item_id'] == sku)), 'status'] = 1
            assign_order_df.to_csv(file_path, index=False)
            self.updated_assigned_order = True
            
            if order.isOrderCompleted():
                self.order_manager.finishOrder(order_id, int(self._tick))
                station = self.station_manager.getStationById(order.station_id)
                station.removeOrder(order_id,order)
                self.insertFinishedOrderToCSV(order)

        job.is_finished = True
        if len(sku_need_replenished) > 0:
            return True
        need_replenish_pod = pod.isNeedReplenishment()
        print(f"reple ga yaaa {need_replenish_pod}")
        return need_replenish_pod
    
    def finishReplenishmentTask(self, job: Job):
        pod: Pod = self.pod_manager.getPodsByCoordinate(job.pod_coordinate.x, job.pod_coordinate.y)
        pod.replenishAllSKU()
        job.is_finished = True
        return False

    def insertFinishedOrderToCSV(self, order: Order):
        header = ["order_id", "order_arrival", "process_start_time", "order_complete_time", "station_id"]
        data = [order.id, order.order_arrival, order.process_start_time, order.order_complete_time,
                order.station_id]

        write_to_csv("order-finished.csv", header, data, self.landscape.current_date_string)

    def findNewOrders(self):
        order_path = os.path.join(PARENT_DIRECTORY, 'data/output/generated_order.csv')
        orders_df = pd.read_csv(order_path)

        file_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
        if os.path.exists(file_path):
            assign_order_df = pd.read_csv(file_path)
        else:
            assign_order_df = orders_df.copy()
            assign_order_df['assigned_station'] = None
            assign_order_df['assigned_pod'] = None
            assign_order_df['status'] = -3
            assign_order_df.to_csv(file_path, index=False)
            self.updated_assigned_order = True
        new_file_df = pd.read_csv(file_path)
                  
        current_second = self.next_process_tick
        previous_second = (self.next_process_tick - 1)

        # Filter orders that have arrived by the current second and have not been processed before
        new_orders = new_file_df[(new_file_df['order_arrival']<= current_second) & 
                               (new_file_df['order_arrival'] > previous_second) &
                               (new_file_df['status'] == -3)]
        grouped_orders = new_orders.groupby('order_id')

        for order_id, group in grouped_orders:
            order_items = group[['item_id', 'item_quantity']].to_dict('records')
            order = self.order_manager.createOrder(order_id, current_second)

            # Add each item in the group to the order
            for item in order_items:
                order.addSKU(item['item_id'], item['item_quantity'])

        return new_orders

    def assignJobToAvailableRobot(self, job: Job):
        current_distance = 1000000
        current_id = -1

        for o in self.getMovableObjects():
            if isinstance(o, Robot) and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                dist = calculate_distance(o.pos_x, o.pos_y, job.pod_coordinate.x, job.pod_coordinate.y)
                if dist < current_distance:
                    current_id = o.id
                    current_distance = dist

        if current_id == -1:
            self.job_queue.append(job)
            return

        for o in self.getMovableObjects():
            if o.id == current_id and isinstance(o, Robot):
                o.assignJobAndSetToTakePod(job)

    def processOrders(self):
        file_path = PARENT_DIRECTORY + "/data/input/assign_order.csv"
        robots_location = []

        # Always read assign_order_df at the start
        assign_order_df = pd.read_csv(
            file_path,
            dtype={
                "sequence_id": "int8",
                "order_id": "int16",
                "order_type": "int8",
                "item_id": "int16",
                "item_quantity": "int8",
                "order_arrival": "int16",
                "assigned_station": "category",
                "assigned_pod": "float32",
                "status": "int8"
            }
        )

        for o in self.getMovableObjects():
            if len(self.job_queue) > 0:
                job: Job = self.job_queue[0]
                if o.object_type == "robot" and (o.job is None or o.job.is_finished) and o.current_state == 'idle':
                    robots_location.append([o.pos_x, o.pos_y])

        for order in self.order_manager.unfinished_orders:
            # If updated_assigned_order is True, reload assign_order_df
            if self.updated_assigned_order:
                assign_order_df = pd.read_csv(
                    file_path,
                    dtype={
                        "sequence_id": "int8",
                        "order_id": "int16",
                        "order_type": "int8",
                        "item_id": "int16",
                        "item_quantity": "int8",
                        "order_arrival": "int16",
                        "assigned_station": "category",
                        "assigned_pod": "float32",
                        "status": "int8"
                    }
                )
                self.updated_assigned_order = False

            if order.station_id is None:
                available_station = self.station_manager.findHighestSimilarityStation(order.skus, self.pod_manager)
                if available_station is not None:
                    order.assignStation(available_station.id)
                    available_station.addOrder(order.id, order)
                    assign_order_df.loc[assign_order_df['order_id'] == order.id, 'assigned_station'] = available_station.id
                    assign_order_df.loc[assign_order_df['order_id'] == order.id, 'status'] = -1
                    assign_order_df.to_csv(file_path, index=False)
                    self.updated_assigned_order = True
                else:
                    break

            if order.process_start_time <= 0:
                order.startProcessing(int(self._tick))

            order_station = self.station_manager.getStationById(order.station_id)
            orders_in_station = order_station.getOrdersInStation()
            skus_in_station = order_station.getSKUsInStation()
            skus_in_station_dict = order_station.getSKUsInStationDict()
            station_coordinate = order_station.coordinate

            for sku in order.getRemainingSKU():
                available_pod: Pod = self.pod_manager.getAvailablePod(sku)
                # For other pod picking strategies, replace the above line

                if available_pod is None:
                    continue
                quantity_to_take = order.getQuantityLeftForSKU(sku)
                order.commitQuantity(sku, quantity_to_take)
                available_pod.pickSKU(sku, quantity_to_take)
                order_station.addPod(available_pod.pod_number)
                available_pod.station = order_station

                assign_order_df.loc[
                    ((assign_order_df['order_id'] == order.id) & (assign_order_df['item_id'] == sku)),
                    'assigned_pod'
                ] = int(available_pod.pod_number)
                assign_order_df.loc[
                    ((assign_order_df['order_id'] == order.id) & (assign_order_df['item_id'] == sku)),
                    'status'
                ] = 0
                assign_order_df.to_csv(file_path, index=False)
                self.updated_assigned_order = True

                job = self.job_manager.createJob(available_pod.coordinate, station_id=order.station_id)
                self.pod_manager.setPodNotAvailable(available_pod.coordinate)
                job.addPickingTask(order.id, sku, quantity_to_take)

                pod_skus = [i for i in available_pod.skus]
                # Turn this off for baseline 
                for skus_pod in pod_skus:
                    for order_ in orders_in_station:
                        if order_ != order and order_.hasSKU(skus_pod):
                            quantity_to_take_other = order_.getQuantityLeftForSKU(skus_pod)
                            if available_pod.getQuantity(skus_pod) > quantity_to_take_other and quantity_to_take_other > 0:
                                order_.commitQuantity(skus_pod, quantity_to_take_other)
                                available_pod.pickSKU(sku, quantity_to_take_other)
                                job.addPickingTask(order_.id, skus_pod, quantity_to_take_other)

                for order_ in orders_in_station:
                    if order_ != order and order_.hasSKU(sku):
                        quantity_to_take_other = order.getQuantityLeftForSKU(sku)
                        if available_pod.getQuantity(sku) > quantity_to_take_other and quantity_to_take > 0:
                            job.addPickingTask(order_.id, sku, quantity_to_take_other)

                self.job_queue.append(job)

    def generateResult(self):
        result = []
        for o in self.getMovableObjects():
            result.append({
                'id': o.id,
                'heading': o.heading,
                'shape': o.shape,
                'velocity': o.velocity,
                'acceleration': o.acceleration,
                'pos_x': o.pos_x,
                'pos_y': o.pos_y,
                'color': o.color,
            })
        return result