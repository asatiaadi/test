import matplotlib.pyplot as plt
import pybullet as p
from time import sleep
import math
import pybullet_data
import numpy as np

class robot_manipulator:
    def __init__(self) :
        p.connect(p.GUI)
        self.robot = p.loadURDF("manipulator/model.urdf", [0,0,0], useFixedBase = 1)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", [0, 0, 0])
        p.setGravity(0, 0, -9.81)

        self.MaxForce = 100
        self.num_joints = p.getNumJoints(self.robot)
        
        self.g = 9.81

        self.m1 = 40
        self.m2 = 40
        self.l1 = 0.70
        self.l2 = 0.70

        #center of mass is at half of the length of the link

        print("{} Parameters {}\n{:^20s}mass_link1 = {} \n{:^20s}mass_link2 = {} ".format('-'*20,'-'*20, ' ', self.m1,' ', self.m2))
        print("{:^20s}length_link1 = {} \n{:^20s}length_link2 = {}".format(' ',self.l1, ' ',self.l2))
        # print("{:^20s}COM_link1 = {} \n{:^20s}COM_link2 = {}".format(' ',self.lc1,' ',self.lc2))
        # print("{:^20s}Inertia_link1 = {} \n{:^20s}Inertia_link2 = {}".format(' ',self.I1,' ',self.I2))



    def moveToCartisian(self, targetpos, targetorn):

        #first the target joint pos are calculated to get using invese kinematics
        tar_j_pos = p.calculateInverseKinematics(bodyUniqueId = self.robot,
                                                        endEffectorLinkIndex= 1,
                                                        targetPosition = targetpos,
                                                        targetOrientation = targetorn) #[0.0, 0.21, 0.0, 0.97])

        # then the joint pos, joint velocity and joint accelaration 
        joint_states = p.getJointStates(self.robot, range(self.num_joints))
        j_pos  = np.around([joint_states[j][0] for j in range(self.num_joints)],decimals = 4)
        j_vel = np.around([joint_states[j][1] for j in range(self.num_joints)],decimals = 4)
        j_acc = np.array([0,0])

        #dt  - time step is defined 
        dt = np.array([0.001,0.001])

        #coefficient for controller
        kt = 10
        kd = 0.0015


        pos_ = []
        vel_ = []
        acc_ = []
        t_ = []
        t=0
        
        tar_j_pos = np.around(tar_j_pos, decimals = 4)
        #Now the equation is itirate for finding joint states
        while(t<40):
            p.stepSimulation()

            #get mass matrix in joint space for dynamics equation of 2DOF
            M = np.around(self.mass_matrix(j_pos), decimals = 3) #mass matrix
            # print(M)

            #get corilis matrix in joint space for dynamics equation of 2DOF
            C = np.around(self.coriolis_matrix(j_pos, j_vel), decimals = 3) #coriolis matrix
            # print(C)
            
            #get gravity matrix in joint space for dynamics equation of 2DOF
            G = np.around(self.gravity_matrix(j_pos), decimals = 3) #gravitational matrix
            # print(G)

            M_inv = np.linalg.inv(M)
            # print(np.shape(M_inv))    

            torq = (kt * (tar_j_pos-j_pos) + kd*(j_vel)) #proportional controller
            # print("torq", torq)
            # Here you can directly measure joint forces from cartesian force
            # torq = k * (target_Joint_torue - current_j_torque)
            # Where current_j_torque can be read from the sensor itself or can be calculated from the dyamics equation  
            
            # substituting all the above values in dynamics equation to get the j_accelaration
            # M.dot(j_acc) +  C.dot(j_vel) + G = torq
            j_acc = M_inv.dot(torq - G - C.dot(j_vel)) 
            
            #converting these values as previous values 
            pre_j_pos = np.around(j_pos, decimals = 4) 
            pre_j_vel = np.around(j_vel, decimals = 4)
            pre_j_acc = np.around(j_acc, decimals = 4)

            #new joint position and joint velocity is calculated
            j_vel = pre_j_vel + pre_j_acc*dt  #d(Vel)/d(t) = acc

            j_pos = pre_j_pos + pre_j_vel*dt  #d(pos)/d(t) = vel

            pos_.append(tar_j_pos - pre_j_pos)
            vel_.append(pre_j_vel)
            acc_.append(pre_j_acc)
            t_.append([t,t])
            sleep(0.001)
            t = t + 0.001
        
        pos_ = np.array(pos_)
        t_ = np.array(t_)
        #graph plot
        # plt.plot( t_[:,0],pos_[:,0], label ="err_j_pos_1")
        # plt.legend(loc = 4)
        # plt.plot(t_[:,0],pos_[:,1], label ="err_j_pos_2")
        # plt.legend(loc = 4)
        # plt.title("Dynamics implementation")
        # plt.xlabel("time step(0.01)")
        # plt.ylabel("theta (rad)")
        # plt.show()    
        # print(cur_j_pos)
        # print(current_j_vel)
        # err_j_pos = 

    def mass_matrix(self, Joint_pos):
        d_11 = (self.m1 * (self.l1)**2 / 3) + (self.m2 * self.l1**2) + (self.m2 * (self.l2**2) / 3) + (self.m2*self.l1*self.l2*np.cos(Joint_pos[1]))
        d_12 = (self.m2 * (self.l2**2) / 3) + (0.5 * self.m2 * self.l1 * self.l2 * np.cos(Joint_pos[1]))
        d_21 = d_12
        d_22 = self.m2 * (self.l2**2) / 3 
        Mass = np.array([[d_11, d_12], #mass matrix
                        [d_21, d_22]]) 
        return(Mass)
    
    def coriolis_matrix(self, Joint_pos, Joint_vel):
        # print(Joint_vel)
        h = - self.m2 * self.l1 * self.l2 * np.sin(Joint_pos[1]) * 0.5            
        c_11 = h * Joint_vel[1]
        c_12 = (h * Joint_vel[1]) + (h * Joint_vel[0])
        c_21 = -(h * Joint_vel[0])
        c_22 = 0
        Coriolis = np.array([[ c_11, c_12 ],
                            [c_21, c_22]])

        return(Coriolis)

    def gravity_matrix(self, Joint_pos):
        g_11 = (0.5 * self.m1 + self.m2) * self.g * self.l1 * np.cos(Joint_pos[0]) + 0.5 * self.m2 * self.g * self.l2 * np.cos(Joint_pos[0] +Joint_pos[1] )
        g_21 = 0.5 * self.m2 * self.l2 * self.g * np.cos(Joint_pos[0] + Joint_pos[1])
        gravity = np.around(np.array([g_11,g_21]), decimals = 3)
        return(gravity)

    def print_curr_j_pos(self):
        joint_states = p.getJointStates(self.robot, range(self.num_joints))
        j_pos_  = np.around([joint_states[j][0] for j in range(self.num_joints)],decimals = 4)
        print(j_pos_)
    
    def robot_(self, theta):

        p.setJointMotorControlArray(bodyUniqueId = self.robot, 
                                        jointIndices = range(2), 
                                        controlMode = p.POSITION_CONTROL, 
                                        targetPositions = theta, 
                                        targetVelocities = [0,0],
                                        forces = [100,100],
                                        positionGains=[0.3,0.3], 
                                        velocityGains=[1,1])

        

 
if __name__ == "__main__" :
    robo = robot_manipulator()

    #robot Target position and orientation
    targetPos = [0.62, 0.0, 0.74]
    targetOrn = [0.0, 0.21, 0.0, 0.97]
    robo.moveToCartisian(targetPos, targetOrn)