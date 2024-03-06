from djitellopy import tello
import time
import cv2
import threading
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import keyboard
# import KeyPressModule as kp

me = tello.Tello()
me.connect()
me.enable_mission_pads()

print(me.get_battery())

folder_name = 'time'
loca_time = str(time.strftime('%m%d_%H%M'))


def video():
    me.send_command_with_return("downvision 1")
    me.streamoff()
    me.streamon()
    frame_read = me.get_frame_read()
    print('Video starts')
    while True:
        
        # get frame
        img = frame_read.frame
        # print(type(img),np.shape(img))
        img = img[:240,:,:]
        # display every frame
        cv2.imshow("Image", img)
        # me.send_command_with_return('command')
        # if cv2.waitKey(5) & 0xFF == ord('q'):
        if cv2.waitKey(5) & 0xFF == 27:  # ASCII('esc') = 27
            me.streamoff()
            break


def imu_read():


    state = me.get_current_state()


    accel_datax = np.array([int(state['agx'])])
    accel_datay = np.array([int(state['agy'])])
    accel_dataz = np.array([int(state['agy'])])

    pitch_data = np.array([int(state['pitch'])])
    roll_data = np.array([int(state['roll'])])
    yaw_data = np.array([int(state['yaw'])])

    speed_datax = np.array([int(state['vgx'])])
    speed_datay = np.array([int(state['vgy'])])
    speed_dataz = np.array([int(state['vgz'])])

    height_data = np.array([int(state['h'])])
    baro_data = np.array([int(state['baro'])])
    time_stamp = np.array([int(time.perf_counter()-t1)])
    # time_stamp = 0


    T1 = time.perf_counter()

    # count = 0
    while True:
        
        state = me.get_current_state()
        
        accel_datax = np.append(accel_datax, int(state['agx']))
        accel_datay = np.append(accel_datay, int(state['agy']))
        accel_dataz = np.append(accel_dataz, int(state['agz']))

        pitch_data = np.append(pitch_data, int(state['pitch']))
        roll_data = np.append(roll_data, int(state['roll']))
        yaw_data = np.append(yaw_data, int(state['yaw']))

        speed_datax = np.append(speed_datax, int(state['vgx']))
        speed_datay = np.append(speed_datay, int(state['vgy']))
        speed_dataz = np.append(speed_dataz, int(state['vgz']))

        height_data = np.append(height_data, int(state['h']))
        time_stamp = np.append(time_stamp, time.perf_counter()-t1)

        # time_stamp = np.append(time_stamp, time.perf_counter() - T1)
        
        time.sleep(0.08)

        # count += 1
        # if count > 15000:
        #     count = 0
            # print(me.send_command_with_return('battery?'),'%')
        if keyboard.is_pressed('esc'):
            # cv2.destroyAllWindows()
            # plt.closel()
            

            T2 = time.perf_counter()
                # print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
            print('帧率:%s帧/s' % (np.shape(accel_datax)[0]/(T2 - T1)))

            
            # imu_data = np.vstack((accel_datax.T,accel_datay.T,accel_dataz.T, pitch_data.T, roll_data.T, yaw_data.T, speed_datax.T, speed_datay.T, speed_dataz.T)).T
            imu_data = np.vstack((accel_datax,accel_datay,accel_dataz+1000, pitch_data, roll_data, yaw_data, speed_datax, speed_datay, speed_dataz, height_data, time_stamp)).T
        
            folder_name = 'data'
            path = os.path.join(folder_name, 'data_'+ loca_time + ".txt")
            np.savetxt(path,imu_data,fmt='%.2f')
            break




def fly():
    me.set_speed(20)
    # count = 0
    while True:

        if keyboard.is_pressed('esc'):
            # me.land()

            break

        if keyboard.is_pressed('space'):
            global T_takeoff
            T_takeoff = time.strftime('%H%M%S')
            addtime('T_takeoff '+ T_takeoff + f' {time.perf_counter() - t1}')
            # me.takeoff()
            me.send_command_with_return('takeoff')
            print('%%%%%%%%%%%%%%%%%%%%%%\n')
            # time.sleep(5)
        if keyboard.is_pressed('h'):
            me.send_command_with_return('rc 0 0 5 0')
            print('#####################\n up 5\n')
        if keyboard.is_pressed('j'):
            me.send_command_with_return('rc 0 0 -5 0')
            print('#####################\n down 5\n')

        if keyboard.is_pressed('up'):
            me.send_command_with_return('forward 20')
            # time.sleep(1)
        if keyboard.is_pressed('down'):
            me.send_command_with_return('back 20')
            # time.sleep(1)

        if keyboard.is_pressed('f'):
            global T_forward
            T_forward = time.strftime('%H%M%S')
            addtime('T_forward '+ T_forward + f' {time.perf_counter() - t1}')
            me.send_command_with_return('forward 200')
            # time.sleep(1)
        if keyboard.is_pressed('b'):
            global T_back
            T_back = time.strftime('%H%M%S')
            addtime('T_back '+ T_back +f' {time.perf_counter() - t1}')
            me.send_command_with_return('back 200')
            # time.sleep(1) wf-=0
        if keyboard.is_pressed('-'):
            global T_onboard
            T_onboard = time.strftime('%H%M%S')
            addtime('T_onboard '+ T_onboard + f' {time.perf_counter() - t1}')
            print('onboard\n')
            # time.sleep(1)
        if keyboard.is_pressed('='):
            global T_offboard
            T_offboard = time.strftime('%H%M%S')
            addtime('T_offboard '+ T_offboard + f' {time.perf_counter() - t1}')
            print('offboard\n')
            # time.sleep(1)
        
        if keyboard.is_pressed('9'):
            
            me.send_command_with_return('forward 50')
            # time.sleep(1)
        if keyboard.is_pressed('0'):
         
            me.send_command_with_return('back 50')
            # time.sleep(1)
        

        if keyboard.is_pressed('w'):
            me.send_command_with_return('up 20')
            # time.sleep(1)
        if keyboard.is_pressed('s'):
            me.send_command_with_return('down 20')
            # time.sleep(1)

        # me.send_rc_control(0, 0, 0, 0)  # 空的移动控制指令，使Tello悬停
        if keyboard.is_pressed('right'):
            me.send_command_with_return('right 20')
            # time.sleep(1)
        if keyboard.is_pressed('left'):
            me.send_command_with_return('left 20')
            # time.sleep(1)

        if keyboard.is_pressed('l'):
            me.land()
            print('land')
            # time.sleep(1)
        if keyboard.is_pressed('q'):
            
            print(me.send_command_with_return('battery?'))
            # time.sleep(1)
        
        time.sleep(0.1)
        # if count == 12000:
        #     me.send_command_with_return('battery?')
        #     count = 0
        # count += 1
        
def addtime(time_info):
    path = os.path.join(folder_name, 'time_'+ loca_time + ".txt")
    with open(path,"a+") as file:
        file.write(time_info+'\n')


def iimmuu():
    T1 = time.perf_counter()
    
    while True:
        state = me.get_current_state()
        print(state)
        # time_stamp = np.append(time_stamp, time.perf_counter() - T1)


if __name__ == '__main__':
    global T_start, t1

    t1 = time.perf_counter()
    imu_thread = threading.Thread(target=imu_read, daemon=True)
    imu_thread = threading.Thread(target=imu_read)
    imu_thread.start()

    # T0 = time.monotonic()
    
    T_start = time.strftime('%H%M%S')
    addtime('T_start '+ T_start + ' 0')
    flyThread = threading.Thread(target=fly, daemon=True)
    flyThread.start()

    # videoThread = threading.Thread(target=video)
    # videoThread.start()

    
    

    # imu_read()
    # iimmuu()
    keyboard.wait('esc')
    me.send_command_with_return('battery?')
    me.send_command_with_return('land')

    T_end = time.strftime('%H%M%S')

   
    # np.savetxt(path, np.array([int(T_start), int(T_takeoff), int(T_forward), int(T_back), int(T_end)]), dtype = int)
    addtime('T_end '+ T_end + f' {time.perf_counter() - t1}')
    time.sleep(2)
    
    # cv2.destroyAllWindows()



