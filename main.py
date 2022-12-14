import pybullet as p
import time
import pybullet_data
import numpy as np
from numpy import pi
from typing import Dict, List


class RGBCamera:
    def __init__(self, cameraEyePosition: np.array, cameraTargetPosition: np.array, cameraUpVector: np.array,
                 fov: float = 45.0, aspect: float = 1.0, nearVal: float = 0.1, farVal: float = 3.1):
        self.cameraEyePosition = cameraEyePosition
        self.cameraTargetPosition = cameraTargetPosition
        self.cameraUpVector = cameraUpVector
        self.fov = fov
        self.aspect = aspect
        self.nearVal = nearVal
        self.farVal = farVal

    def compute_view_and_projection_matrix(self):
        self.view_matrix = p.computeViewMatrix(
            self.cameraEyePosition,
            self.cameraTargetPosition,
            self.cameraUpVector
        )
        self.projection_matrix = p.computeProjectionMatrix(
            self.fov,
            self.aspect,
            self.nearVal,
            self.farVal
        )

    def get_image(self):
        width, height, rgbImg, depthImg, segImg = p.getCameraImage(
            width=224,
            height=224,
            viewMatrix=self.view_matrix,
            projectionMatrix=self.projection_matrix)
        return width, height, depthImg, segImg


class R2D2:
    """
    This class represents a R2D2 robot in our pybullet simulation.
    :param start_pos: Starting position of R2D2 as 3-dimensional vector
    :type start_pos: np.ndarray
    :param start_orientation: Starting orientation of R2D2 in euler angles as a 3 dimensional vector
    :type start_orientation: np.ndarray
    """

    def __init__(self, start_pos: np.ndarray, start_orientation: np.ndarray):
        """
        Constructor method.
        """
        # transform the orientation from euler angles to quaternions
        start_orientation_quat = np.array(p.getQuaternionFromEuler(start_orientation))
        # load in the robot with a given starting position and orientation
        self.object_ID = p.loadURDF("r2d2.urdf", start_pos, start_orientation_quat)

        # initialize the current and last position and orientation of R2D2 to the input values
        self.current_position = start_pos
        self.last_position = start_pos
        self.current_orientation = start_orientation
        self.last_orientation = start_orientation

    def get_joint_info(self):
        """
        Returns information about the joints of the R2D2 robot as specified in the URDF.
        :return: Joint information of R2D2 robot
        :rtype: Dict
        """
        joint_info = {}
        for i in range(p.getNumJoints(self.object_ID)):
            joint_info_list = p.getJointInfo(self.object_ID, i)
            joint_info[joint_info_list[0]] = {
                "jointName": joint_info_list[1],
                "jointType": joint_info_list[2],
                "qIndex": joint_info_list[3],
                "uIndex": joint_info_list[4],
                "flags": joint_info_list[5],
                "jointDamping": joint_info_list[6],
                "jointFriction": joint_info_list[7],
                "jointLowerLimit": joint_info_list[8],
                "jointUpperLimit": joint_info_list[9],
                "jointMaxForce": joint_info_list[10],
                "jointMaxVelocity": joint_info_list[11]
            }
        return joint_info

    def get_position_and_orientation(self):
        """
        Return the position and the orientation of R2D2.
        :return: Position of R2D2 in cartesian coordinates and its orientation in euler angles as multiples of pi
        :rtype: List[np.array, np.array]
        """
        pos, orn = p.getBasePositionAndOrientation(self.object_ID)
        return np.array(pos), (np.array(p.getEulerFromQuaternion(orn)) / pi)

    def drive(self, force: float = 100, velocity: float = 20):
        """
        Make R2D2 accelerate with a given force to a given velocity. Once they reach the velocity they move
        at a constant speed.
        Make R2D2 drive with a given velocity and a force with which it accelerates to the given velocity.
        :param force: Acceleration force, defaults to 100
        :type force: float
        :param velocity: Target velocity
        :type velocity: float
        :return: None
        """
        velocity = -velocity
        p.setJointMotorControlArray(self.object_ID,
                                    jointIndices=[2, 3, 6, 7],
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=np.repeat(force, 4),
                                    targetVelocities=np.repeat(velocity, 4))

    def stop(self, force):
        """
        Stops R2D2 with a given force.
        :param force: Breaking force
        :return: None
        """
        p.setJointMotorControlArray(self.object_ID,
                                    jointIndices=[2, 3, 6, 7],
                                    controlMode=p.VELOCITY_CONTROL,
                                    forces=np.repeat(force, 4),
                                    targetVelocities=np.repeat(0, 4))

    def update_position_and_orientation_data(self):
        curr_pos, curr_orn = self.get_position_and_orientation()
        if (curr_pos != self.current_position).any():
            self.last_position = self.current_position
            self.current_position = curr_pos

        if (curr_orn != self.current_orientation).any():
            self.last_orientation = self.current_orientation
            self.current_orientation = curr_orn


def main():
    # connect to GUI built-in physics server
    # physicsClient = p.connect(p.GUI)
    p.connect(p.GUI)

    # above we imported some data that comes with the package installation, which we now
    # set as an additional search path when searching for data. This is optional.
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # By default, a pybullet environment has no gravitational force enabled
    # Here we enable a gravitational force of -10m/s^2 along the z-axis
    p.setGravity(gravX=0, gravY=0, gravZ=-10)

    # the loadURDF function loads a physics model from a Universal Robot
    # Description File (URDF). In the case below we are loading in a two-dimensional plane
    # along the x- and y-axis
    planeId = p.loadURDF("plane.urdf")

    # We define the start positions of R2D2 in cartesian coordinates
    startPos = np.array([1, 1, 1])

    # Define the start orientation of R2D2 in euler angles
    startOrientation = np.array([0, 0, pi * 3 / 4])

    # Construct R2D2
    r2d2 = R2D2(startPos, startOrientation)

    # Add the camera
    rgb_camera = RGBCamera(
        view_matrix=p.computeViewMatrix(
            cameraEyePosition=[0, 0, 3],
            cameraTargetPosition=[0, 0, 0],
            cameraUpVector=[0, 1, 0]),
        projection_matrix=p.computeProjectionMatrixFOV(
            fov=45.0,
            aspect=1.0,
            nearVal=0.1,
            farVal=3.1)
    )
    # # Print joint info
    # joint_info = r2d2.get_joint_info()
    # for key in r2d2.get_joint_info():
    #     print(f"Joint {key}: {joint_info[key]}")

    # start the simulation
    for i in range(10000):
        # this allows to drag the robot around in the GUI using forward dynamics
        p.stepSimulation()
        # allow collision detection
        p.performCollisionDetection()

        if i == 0:
            r2d2.drive(force=100, velocity=20)

        if i == 1000:
            r2d2.stop(20)

        if i % 100 == 0:
            rgb_camera.get_image()

        r2d2.update_position_and_orientation_data()
        print("Position and orientation by method")
        print(r2d2.get_position_and_orientation())
        print("Position and orientation by attribute")
        print(r2d2.current_position, r2d2.current_orientation)
        print(r2d2.last_position, r2d2.current_position)
        time.sleep(1. / 240.)
    # End the simulation
    p.disconnect()


if __name__ == "__main__":
    main()
