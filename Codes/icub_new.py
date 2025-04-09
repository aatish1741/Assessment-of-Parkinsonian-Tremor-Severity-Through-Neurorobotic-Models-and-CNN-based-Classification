#!/usr/bin/env python3
import yarp
import time
import numpy as np
import socket
import errno
from socket import error as SocketError
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq
import select
import argparse
import csv

class Movement():
    def __init__(self, mode='healthy'):
        self.mode = mode  # mode can be 'healthy' or 'pd'
        print("Running in mode:", self.mode)
        # Initialize YARP and open the iCub controlboard
        yarp.Network.init()
        props = yarp.Property()
        props.put("device", "remote_controlboard")
        props.put("local", "/client/left_arm")
        props.put("remote", "/icubSim/left_arm")
        armDriver = yarp.PolyDriver(props)
        self.iPos = armDriver.viewIPositionControl()
        self.iVel = armDriver.viewIVelocityControl()
        self.iTorque = armDriver.viewITorqueControl()
        self.iEnc = armDriver.viewIEncoders()
        self.iimp = armDriver.viewIImpedanceControl()
        self.ictrl = armDriver.viewIControlMode()
        self.jointsConn()
        self.setSpeed()
        # Lists to store time series data for CSV output and plotting
        self.joint4_pos = []
        self.joint5_pos = []
        self.t = []
        # Start socket connection to the computational model (MarmosetBG)
        self.socketreceive()
        # Run the movement sequence
        self.MakeMovement()
        # After movement is finished, save CSV and plot the time series
        self.save_csv()
        self.plot_time_series()

    def jointsConn(self):
        # Retrieve number of joints
        self.jnts = self.iPos.getAxes()
        print('Controlling', self.jnts, 'joints')
        self.encs = yarp.Vector(self.jnts)
        self.iEnc.getEncoders(self.encs.data())

    def setSpeed(self):
        self.iPos.setRefSpeed(1, 10)
        self.iPos.setRefAcceleration(1, 50)
        self.iPos.setRefSpeed(4, 180)
        self.iPos.setRefAcceleration(4, 60)
        self.iPos.setRefSpeed(5, 50)
        self.iPos.setRefAcceleration(5, 20)
        # Define the target positions for pronation and supination
        self.prono_pos = yarp.Vector(self.jnts)
        self.prono_pos.set(0, -23.1)
        self.prono_pos.set(1, 24)
        self.prono_pos.set(2, 14.04)
        self.prono_pos.set(3, 64.61)
        self.prono_pos.set(4, 60)
        self.supin_pos = yarp.Vector(self.jnts)
        self.supin_pos.set(0, -23.1)
        self.supin_pos.set(1, 16)
        self.supin_pos.set(2, 14.04)
        self.supin_pos.set(3, 64.61)
        self.supin_pos.set(4, -60)
        # Record start time and initialize sampling times
        self.start = time.time()
        self.last_samp_t = self.start
        self.last_trem_t = self.start
        self.tremor_state = 0
        self.tremor_total = []
        # For now, you could later adjust PD_state based on mode
        # For example, if mode=='healthy', force PD_state=0; if 'pd', use the received value.
        self.PD_state = 0 if self.mode == 'healthy' else 1

    def generate_movement(self):
        # Run the movement for a fixed duration (here 9000*0.01 seconds)
        while time.time() < self.start + 9000 * 0.01:
            # Check if there is data on the socket
            socket_as_list = [self.coeff]
            readable, _, _ = select.select(socket_as_list, [], [], 0.1)
            if self.coeff in readable:
                try:
                    # Compute tremor state from collected data (if any)
                    self.tremor_state = np.array(self.tremor_total).mean() if self.tremor_total else 0
                    self.tremor_total = []
                    self.coeff.send(str(self.tremor_state).encode())
                    print('TREMOR MEAN SENT:', self.tremor_state)
                    self.tremor_state = 0
                    # Receive PD_state value from the computational model
                    self.PD_state = float(self.coeff.recv(1024).decode())
                    print('PD COEFF RECEIVED:', self.PD_state)
                except (socket.error, ValueError) as e:
                    print('Error receiving data:', e)
            print('Time:', time.time() - self.start)
            # Move to pronation position
            print('Moving pronation position')
            self.iPos.positionMove(self.prono_pos.data())
            while not self.checkReachPos('sup', None, 55, 4):
                self.check_t = time.time()
                if self.check_t - self.last_samp_t >= 0.01:
                    self.joint4_pos.append(self.getPos(4))
                    self.joint5_pos.append(self.getPos(5))
                    self.tremor_total.append(self.getPos(5))
                    self.t.append(self.check_t - self.start)
                    self.last_samp_t = self.check_t
                # Optionally, if PD_state > 1, generate tremor
                if self.PD_state > 1:
                    if self.check_t - self.last_trem_t >= 0.2:
                        trem_amp = 2.5 / 8 * self.PD_state + 0.2
                        self.generateSingleTremor(trem_amp, 5)
                        self.last_trem_t = self.check_t
            # Move to supination position
            print('Moving supination position')
            self.iPos.positionMove(self.supin_pos.data())
            while not self.checkReachPos('inf', -55, None, 4):
                self.check_t = time.time()
                if self.check_t - self.last_samp_t >= 0.01:
                    self.joint4_pos.append(self.getPos(4))
                    self.joint5_pos.append(self.getPos(5))
                    self.tremor_total.append(self.getPos(5))
                    self.t.append(self.check_t - self.start)
                    self.last_samp_t = self.check_t
                if self.PD_state > 1:
                    if self.check_t - self.last_trem_t >= 0.01:
                        trem_amp = 2.2 / 8 * self.PD_state + 0.2
                        self.generateSingleTremor(trem_amp, 5)
                        self.last_trem_t = self.check_t
        print('Total time points recorded:', len(self.t), 'Joint4 samples:', len(self.joint4_pos))

    def socketreceive(self):
        # Connect to the computational model running on the MarmosetBG side
        self.coeff = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_host = 'localhost'
        receiver_port = 12345 
        self.coeff.connect((server_host, receiver_port))
        print('Receiver socket connected')

    def StartPos(self):
        self.startpos = yarp.Vector(self.jnts)
        self.startpos.set(0, -23.1)
        self.startpos.set(1, 24)
        self.startpos.set(2, 14.04)
        self.startpos.set(3, 64.61)
        self.startpos.set(4, 60)
        self.iPos.positionMove(self.startpos.data())
        while not self.checkReachPos('sup', None, 58, 4):
            print('Going to start position...')
        print('Start position (pronation) reached, beginning movement in 3s')
        time.sleep(3)

    def getPos(self, jnt):
        self.iEnc.getEncoders(self.encs.data())
        return yarp.Vector(jnt+1, self.encs.data())[jnt]

    def checkReachPos(self, comp, boundary_inf, boundary_sup, jnt):
        if comp == 'inf':
            if self.getPos(jnt) <= boundary_inf:
                return True
        elif comp == 'sup':
            if self.getPos(jnt) >= boundary_sup:
                return True
        elif comp == 'between':
            if self.getPos(jnt) <= boundary_inf and self.getPos(jnt) >= boundary_sup:
                return True
        return False

    def generateSingleTremor(self, amp, jnt):
        # "Up" phase of tremor
        self.iPos.positionMove(jnt, amp)
        while not self.checkReachPos('sup', None, amp * 0.8, jnt):
            check_t = time.time()
            if check_t - self.last_samp_t >= 0.01:
                self.joint4_pos.append(self.getPos(4))
                self.joint5_pos.append(self.getPos(5))
                self.tremor_total.append(self.getPos(5))
                self.t.append(check_t - self.start)
                self.last_samp_t = check_t
        # "Down" phase of tremor
        self.iPos.positionMove(jnt, -amp)
        while not self.checkReachPos('inf', -amp * 0.8, None, jnt):
            check_t = time.time()
            if check_t - self.last_samp_t >= 0.01:
                self.joint4_pos.append(self.getPos(4))
                self.joint5_pos.append(self.getPos(5))
                self.tremor_total.append(self.getPos(5))
                self.t.append(check_t - self.start)
                self.last_samp_t = check_t
        # Return joint back to 0
        self.iPos.positionMove(jnt, 0)

    def save_csv(self):
        # Save the recorded time series to a CSV file; the filename depends on the mode.
        filename = f"icub_{self.mode}_movement.csv"
        print(f"Saving CSV file as {filename}")
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['time', 'joint4', 'joint5']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for t_val, j4, j5 in zip(self.t, self.joint4_pos, self.joint5_pos):
                writer.writerow({'time': t_val, 'joint4': j4, 'joint5': j5})

    def plot_time_series(self):
        # Generate a plot of joint4 and joint5 time series, then save as PNG.
        plt.figure(figsize=(10, 5))
        plt.plot(self.t, self.joint4_pos, label='Joint 4')
        plt.plot(self.t, self.joint5_pos, label='Joint 5')
        plt.xlabel('Time (s)')
        plt.ylabel('Joint Angle')
        plt.title(f"ICub {self.mode.capitalize()} Movement Time Series")
        plt.legend()
        out_filename = f"icub_{self.mode}_timeseries.png"
        plt.savefig(out_filename)
        print(f"Saved time series plot as {out_filename}")
        plt.show()

    def MakeMovement(self):
        # Reset the lists for each run
        self.joint4_pos = []
        self.joint5_pos = []
        self.t = []
        self.socketreceive()
        self.StartPos()
        self.generate_movement()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run iCub movement and generate CSV/PNG outputs")
    parser.add_argument('--mode', type=str, default='healthy', choices=['healthy', 'pd'], 
                        help="Specify the mode: 'healthy' or 'pd'")
    args = parser.parse_args()
    Move = Movement(mode=args.mode)
