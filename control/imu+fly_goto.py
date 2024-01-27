import logging
import time
import numpy as np
import cflib.crtp
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from cflib.crazyflie.log import LogConfig
import keyboard
import datetime
# 获取当前日期和时间
current_datetime = datetime.datetime.now()
# 将日期和时间格式化为字符串，例如：2023-11-01_12
datetime_string = current_datetime.strftime("%m%d-%H%M")


URI = 'radio://0/80/2M/E7E7E7E707'

# Only output errors from the logging framework
logging.basicConfig(level=logging.ERROR)

# Create an empty NumPy array to hold the IMU data
imu_data = np.empty((0, 13))

def process_imu_data(timestamp, data, logconf):
    global imu_data
    acc_x = data['stateEstimateZ.ax']
    acc_y = data['stateEstimateZ.ay']
    acc_z = data['stateEstimateZ.az']
    gyro_x = data['gyro.xRaw']
    gyro_y = data['gyro.yRaw']
    gyro_z = data['gyro.zRaw']

    # gyro_x = data['stateEstimateZ.ratePitch']
    # gyro_y = data['stateEstimateZ.rateRoll']
    # gyro_z = data['stateEstimateZ.rateYaw']

    motor1 = data['motor.m1s']
    motor2 = data['motor.m2s']
    motor3 = data['motor.m3s']
    motor4 = data['motor.m4s']
    quat = data['stateEstimateZ.quat']


    current_time = time.monotonic() - start_time
    height = data['range.zrange']
    # print(f"IMU data: acc_x={acc_x}, acc_y={acc_y}, acc_z={acc_z}, gyro_x={gyro_x}, gyro_y={gyro_y}, gyro_z={gyro_z},time={current_time}")
    imu_data = np.vstack((imu_data, [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z, motor1, motor2, motor3, motor4, quat, current_time, height]))




if __name__ == '__main__':
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)

    with SyncCrazyflie(URI) as scf:
        cf = scf.cf

        # Add IMU log config version 1
        # imu_log_config = LogConfig(name='imu_data', period_in_ms=10)
        # imu_log_config.add_variable('acc.x', 'float')
        # imu_log_config.add_variable('acc.y', 'float')
        # imu_log_config.add_variable('acc.z', 'float')
        # imu_log_config.add_variable('gyro.x', 'float')
        # imu_log_config.add_variable('gyro.y', 'float')
        # imu_log_config.add_variable('gyro.z', 'float')
        # imu_log_config.add_variable('range.zrange', 'uint16_t')



        # Add IMU log config version 1
        imu_log_config = LogConfig(name='imu_data', period_in_ms=10)
        imu_log_config.add_variable('stateEstimateZ.ax', 'int16_t')
        imu_log_config.add_variable('stateEstimateZ.ay', 'int16_t')
        imu_log_config.add_variable('stateEstimateZ.az', 'int16_t')
        imu_log_config.add_variable('gyro.xRaw', 'int16_t')
        imu_log_config.add_variable('gyro.yRaw', 'int16_t')
        imu_log_config.add_variable('gyro.zRaw', 'int16_t')
        imu_log_config.add_variable('motor.m1s', 'uint16_t')
        imu_log_config.add_variable('motor.m2s', 'uint16_t')
        imu_log_config.add_variable('motor.m3s', 'uint16_t')
        imu_log_config.add_variable('motor.m4s', 'uint16_t')
        imu_log_config.add_variable('stateEstimateZ.quat', 'uint32_t')
        imu_log_config.add_variable('range.zrange', 'uint16_t')


        cf.log.add_config(imu_log_config)
        start_time = time.monotonic()
        imu_log_config.data_received_cb.add_callback(process_imu_data)
        imu_log_config.start()
        # Wait for the user to be ready to take off
        # input('Press Enter when ready to take off...')
        # time.sleep(10)
        # np.savetxt('imu_data.csv', imu_data, delimiter=',')
        # Loop until ESC is pressed
        # print('Press space when ready to take off...')
        input('Press space when ready to take off...')
        # if keyboard.wait('t'):
        mc = MotionCommander(scf)
        mc.take_off()
        print('Taking off!')
        try:
            while True:
                # Check for keyboard input
                # print('round')

                if keyboard.is_pressed('up'):
                    # Move forward
                    print('Moving forward 0.1m')
                    mc.forward(0.1)
                elif keyboard.is_pressed('down'):
                    # Move backward
                    print('Moving backward 0.1m')
                    mc.back(0.1)
                elif keyboard.is_pressed('f'):
                    # Move forward
                    print('Moving forward 2m')
                    mc.forward(2,0.2)
                elif keyboard.is_pressed('b'):
                    # Move backward
                    print('Moving backward 2m')
                    mc.back(2,0.2)
                elif keyboard.is_pressed('left'):
                    # Move left
                    print('Moving left 0.1m')
                    mc.left(0.1)
                elif keyboard.is_pressed('right'):
                    # Move right
                    print('Moving right 0.1m')
                    mc.right(0.1)
                elif keyboard.is_pressed('w'):
                    # Move up
                    print('Moving up 0.1m')
                    mc.up(0.1)
                elif keyboard.is_pressed('6'):
                    # Move up
                    print('Moving up 0.06m')
                    mc.up(0.06)
                elif keyboard.is_pressed('s'):
                    # Move down
                    print('Moving down 0.1m')
                    mc.down(0.1)
                elif keyboard.is_pressed('a'):
                    # Rotate left
                    print('Rotating left')
                    mc.turn_left(10)
                elif keyboard.is_pressed('d'):
                    # Rotate right
                    print('Rotating right')
                    mc.turn_right(10)
                elif keyboard.is_pressed(' '):
                    # Hover in place
                    print('Hovering')
                    mc.hove()
                elif keyboard.is_pressed('o'):

                    flight_time = 2.0

                    commander = scf.cf.high_level_commander

                    commander.go_to(0, 0, 1.2, 0, flight_time, relative=False)
                    time.sleep(flight_time)

                elif keyboard.is_pressed('p'):
                    flight_time = 8.0

                    commander = scf.cf.high_level_commander

                    commander.go_to(1.5, 0, 1.2, 0, flight_time, relative=False)
                    time.sleep(flight_time)


                # Check for ESC key press
                if keyboard.is_pressed('esc'):
                    # Land and stop the MotionCommander object
                    print('Landing!')
                    mc.land()
                    # mc.stop()
                    imu_log_config.stop()
                    
                    # Save the IMU data to a file
                    np.savetxt('/home/nuci7/project/cf2/crazyflie-firmware/control/data/imu_data_'+datetime_string+'.csv', imu_data, delimiter=',')
                    break

                # Wait a bit before checking for input again
                time.sleep(0.02)

        except:
            np.savetxt('/home/nuci7/project/cf2/crazyflie-firmware/control/data/imu_data_'+datetime_string+'.csv', imu_data, delimiter=',')
