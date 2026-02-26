import math
import numpy as np
from pydrake.all import (Solve, RotationMatrix, JacobianWrtVariable, RigidTransform, InverseKinematics)
from dataclasses import dataclass
from .trajectory_models import BallBouncingTrajectory
from .controller_drake_setup import DrakeSceneHelper
from scipy.spatial.transform import Rotation as R

@dataclass()
class RobotState:
    time_system_offset: float
    time: float
    q: list[float]
    X_WE: RigidTransform
    v_WE: list[float]

class HitScorer():
    
    def __init__(self, plant, plant_context, frame_B, position_offset_B):
        self.plant = plant
        self.plant_context = plant_context
        self.frame_B = frame_B
        self.position_offset_B = position_offset_B
    
    def score(self, current_pose = None, change_weight = None, time = None):
        return self.yoshikawa_manipulability_score() / ((self.change_score(current_pose)**2) * (time or 1)**2)
        
    def yoshikawa_manipulability_score(self):
                
        jacobian = self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context,
            JacobianWrtVariable.kV,
            self.frame_B,
            self.position_offset_B,
            self.plant.world_frame(),
            self.plant.world_frame()
        )
        
        jacobian = jacobian[:2] # 2D only
        determinant = np.linalg.det(jacobian @ jacobian.T)
        if (determinant < 0): return 0
        return math.sqrt(determinant)
    
    def change_score(self, current_pose, change_weight = 1):
        if current_pose is None or change_weight is None: return 1
        q_new = self.plant.GetPositions(self.plant_context)
        l2_distance = np.linalg.norm(q_new - np.array(current_pose))
        return l2_distance * change_weight
    
class HitBruteSearcher():
    
    def __init__(self, h: DrakeSceneHelper, num_samples: int = 10, debug: bool = False):
        self.h = h
        self.num_samples = num_samples
        self.debug = debug
        self.scorer = HitScorer(
            self.h.plant,
            self.h.plant_context,
            self.h.plant.GetFrameByName("end_effector", self.h.robot),
            [0,0,0]
        )
        
        self.__find_dimensions()
        self.__initialize_prog()

    def find_best_hit(self,
                      traj: BallBouncingTrajectory,
                      traj_offset_time, 
                      p_WTarget,
                      last_hit_state: RobotState = None,
        ) -> RobotState:
        
        time_range = find_time_range_within_semi_circle(traj, self.robot_origin, self.robot_radius)
        if len(time_range) != 2: return None
        sample_times =  np.linspace(time_range[0], time_range[1], self.num_samples)
          
        initial_guess = [math.pi/4, 0, 0]
        best_score = 0
        for t in sample_times:
                      
            p_WBallhit = traj.calculate(t)
            if not self.__is_ball_within_z_reach(p_WBallhit[2]):
                continue
                
            v_WBallhit = traj.calculate_v(t)
            bisect = find_2D_paddle_orientation(p_WBallhit, p_WTarget, v_WBallhit)
  
            for forehand in [-1, 1]:
            
                rot, _ = R.align_vectors(normalize(forehand * bisect), [0,1,0])
                paddle_orientation = RotationMatrix(rot.as_matrix())

                for _ in range(3):
                    
                    result = self.__solve_ik_pose(p_WBallhit, paddle_orientation, initial_guess)
                    success = result.is_success()
                    if success: break
                    np.random.seed(1)
                    initial_guess = np.random.rand(3) * math.pi/6

                if not success: continue
                
                initial_guess = result.GetSolution(self.ikprog.q())
                if last_hit_state:
                    current_pose = last_hit_state.q
                else:
                    current_pose = None
                score = self.scorer.score(current_pose, 1, t)

                if score > best_score:
                    best_score = score
                    state = RobotState(
                        traj_offset_time, # this will become the offset given by the trajectory topic (time photo taken)
                        t,
                        result.GetSolution(self.ikprog.q()),
                        RigidTransform(paddle_orientation, p_WBallhit),
                        bisect
                    )
            
        if best_score != 0:
            return state

        return None
        
        
    def __initialize_prog(self):
        
        self.ikprog = InverseKinematics(self.h.plant, self.h.plant_context)
        self.prog = self.ikprog.prog()
        self.q = self.ikprog.q()
        
        bound = [0,0,0]
        placeholder_pose = RigidTransform([.2, -.2, 0])
        q0 = [-.01, .01, .01]
    
        self.position_constraint = self.ikprog.AddPositionConstraint(
            self.end,
            [0.08, 0, 0],
            self.w,
            [0,0,0],
            [0,0,0]
        )
        
        rot, _ = R.align_vectors([1,0,0], [1,0,0])
        goal = RotationMatrix(rot.as_matrix())
        self.orientation_constraint = self.ikprog.AddOrientationConstraint(
            self.end,
            RotationMatrix(),
            self.w,
            goal,
            .1,
        )
                
        self.prog.SetInitialGuess(self.q, q0)
        self.prog.AddQuadraticErrorCost(np.identity(len(self.q)), q0, self.q)
    
    
    def __find_dimensions(self):
        
        base = self.h.plant.GetFrameByName("base_link", self.h.robot)
        end = self.h.plant.GetFrameByName("end_effector", self.h.robot)
        w = self.h.plant.world_frame()

        self.h.plant.SetPositions(self.h.plant_context, self.h.robot, [0,0,0])

        X_WB = base.CalcPose(self.h.plant_context, w)
        p_WB = X_WB.translation()
        
        X_WE = end.CalcPose(self.h.plant_context, w)
        p_WE = X_WE.translation()
        
        self.end = end
        self.w = w
        self.robot_origin = p_WB[:2]
        self.robot_radius = max(p_WE[:2]) # + .08 # approximate
        self.robot_paddle_height = p_WE[2]
        self.robot_paddle_radius = .06
        
        
    def __solve_ik_pose(self, pose, orientation, initial_guess = None):
        
        bound = np.array([0.01,0.01,self.robot_paddle_radius])
        self.position_constraint.evaluator().set_bounds(pose - bound, pose + bound)
        
        self.prog.RemoveConstraint(self.orientation_constraint)
        self.orientation_constraint = self.ikprog.AddOrientationConstraint(
            self.end,
            RotationMatrix(),
            self.w,
            orientation,
            .01,
        )
        
        if initial_guess is not None:
            self.prog.SetInitialGuess(self.ikprog.q(), initial_guess)
            
        result = Solve(self.prog)
        return result
        
        
    def __is_ball_within_z_reach(self, z: float):
        return self.robot_paddle_height - self.robot_paddle_radius < z < self.robot_paddle_height + self.robot_paddle_radius
    
def normalize(v):
    return v/(np.linalg.norm(v))
    
def find_time_range_within_circle(t: BallBouncingTrajectory, origin, radius):
    if t.x_vel == 0: return np.array([])
    a = (1 + ((t.y_vel**2)/(t.x_vel**2)))
    u = (t.y_vel*t.x_pos)/t.x_vel - t.y_pos + origin[1]
    b = -2*(origin[0] + (u*t.y_vel/t.x_vel))
    c = origin[0]**2 + u**2 - radius**2
    x_range = quadratic_formula(a, b, c)
    time_range = (x_range - t.x_pos) / t.x_vel
    time_range.sort()
    return time_range

def find_time_range_within_semi_circle(t: BallBouncingTrajectory, origin, radius):
    if t.x_vel == 0: return np.array([])
    a = (1 + ((t.y_vel**2)/(t.x_vel**2)))
    u = (t.y_vel*t.x_pos)/t.x_vel - t.y_pos + origin[1]
    b = -2*(origin[0] + (u*t.y_vel/t.x_vel))
    c = origin[0]**2 + u**2 - radius**2
    x_range = quadratic_formula(a, b, c)
    if all(x < 0 for x in x_range): return np.array([]) # this includes empty x_range
    x_range_semi = np.array([max(x,0) for x in x_range])
    time_range = (x_range_semi - t.x_pos) / t.x_vel
    time_range.sort()
    return time_range


def quadratic_formula(a, b, c):
    discriminant = b**2 - (4*a*c)
    if discriminant < 0: return np.array([])
    s1 = (-b - math.sqrt(discriminant)) / (2 * a)
    if discriminant == 0: return np.array([s1])
    s2 = (-b + math.sqrt(discriminant)) / (2 * a)
    return np.array([s1, s2])

def find_2D_paddle_orientation(p_WBallhit, p_WTarget, v_WBallhit):
    p_WBallhit_c = p_WBallhit.copy()
    p_WTarget_c = np.array(p_WTarget).copy()
    p_WBallhit_c[-1] = 0
    p_WTarget_c[-1] = 0
    p_TargetBallhit = -p_WTarget_c + p_WBallhit_c
    bisect = normalize(p_TargetBallhit) + normalize(v_WBallhit)
    return bisect