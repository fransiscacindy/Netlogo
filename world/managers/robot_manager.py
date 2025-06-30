from __future__ import annotations
from typing import List, TYPE_CHECKING
from world.entities.robot import Robot
from ai.MAA2C import MAA2C

if TYPE_CHECKING:
    from world.warehouse import Warehouse

class RobotManager:
    def __init__(self, warehouse: Warehouse):
        self.warehouse = warehouse
        self.robots: List[Robot] = []
        self.robot_counter = 0
        
        self.heuristic_rl = False
        self.current_rl_state = {}
        self.previous_rl_state = {}
    
    def initRobotManager(self):
        for robot in self.robots:
            robot.setRobotManager(self)
    
    def getAllRobots(self):
        return self.robots
    
    def getRobotByName(self, robot_name):
        for o in self.getAllRobots():
            if o.object_type == "robot" and o.robotName() == robot_name:
                return o

    def getRobotByCoordinate(self, x, y):
        for o in self.getAllRobots():
            if o.object_type == "robot" and o.pos_x == x and o.pos_y == y:
                return o
                    
    def getRobotsByCoordinate(self, coords):
        robots = []
        for coord in coords:
            robot = self.getRobotByCoordinate(coord[0], coord[1])
            if robot:
                robots.append(robot)
        return robots
    
    def createRobot(self, x, y):
        robot = Robot(self.robot_counter, x, y)
        self.robots.append(robot)
        self.robot_counter += 1
        robot._id = self.warehouse.total_pod + 1
        self.warehouse.total_pod += 1
        return robot
    
    def createNewModel(self, robot: Robot, state):
        state_size = len(state)
        return MAA2C(state_size=state_size, action_size=len(self.get_action_space()), model_name=robot.RL_model_name)

    def updateDirectionUsingDQN(self, tick):
        for robot in self.robots:  # Loop through each robot instead of intersections
            if robot.use_reinforcement_learning:
                self.handleModel(robot, tick)

    def updateModelAfterExecution(self, tick):
        for robot in self.robots:
            if robot.use_reinforcement_learning and robot.RL_model_name in self.q_models:
                self.rememberAndReplay(robot, self.calculateReward(robot, tick),
                                    self.isEpisodeDone(robot, tick), tick)

    def rememberAndReplay(self, robot: Robot, reward, done, tick):
        model = self.q_models[robot.RL_model_name]
        
        if robot.id in self.previous_state and robot.id in self.previous_action:
            next_state = self.getState(robot, tick)
            model.remember(self.previous_state[robot.id],
                        self.previous_action[robot.id], reward, next_state, done)
            if done:
                model.replay(64)

            self.resetStateAction(robot)

        if tick % 1000 == 0 and tick != 0:
            print("SAVING_MODEL")
            robot.resetTotals()
            model.save_model(robot.RL_model_name, tick)

    def resetStateAction(self, robot: Robot):
        if robot.id in self.previous_state:
            del self.previous_state[robot.id]
        if robot.id in self.previous_action:
            del self.previous_action[robot.id]

    @staticmethod
    def isEpisodeDone(robot: Robot, tick):
        if robot.hasReachedDestination():
            return True
        elif int(tick) % 1000 == 0:
            return True
        else:
            return False

    def calculateReward(self, robot: Robot, tick):
        reward = 0

        if robot.hasPassedIntersection():
            reward += self.calculatePassingRobotReward(robot, tick)

        reward += self.calculateCurrentRobotReward(robot, tick)

        if robot.isIdle():
            reward += -0.1  # Penalize for staying idle too long

        return reward
    
    def get_action_space(self):
        return [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]

