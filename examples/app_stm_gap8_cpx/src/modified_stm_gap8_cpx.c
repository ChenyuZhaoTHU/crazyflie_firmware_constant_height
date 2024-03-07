/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--´  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2023 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * App layer application that communicates with the GAP8 on an AI deck.
 */

/*
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
*/



#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"

#include "cpx.h"
#include "cpx_internal_router.h"

#include "FreeRTOS.h"
#include "task.h"
#include "log.h"
#define DEBUG_MODULE "APP"
#include "debug.h"

// Callback that is called when a CPX packet arrives
static void cpxPacketCallback(const CPXPacket_t* cpxRx);

static CPXPacket_t txPacket;

#define mass 0.033 // mass
#define g 9.81 // gravity
#define rho 1.225 // air density
#define D 0.05  // diameter of the rotor
#define rpm2rs 3600
#define C_t_fixed 0.09937873049125099



int16_t accZ = 0;
uint16_t motor1 = 0;
uint16_t motor2 = 0;
uint16_t motor3 = 0;
uint16_t motor4 = 0;
uint32_t r_square = 0;
float thrust = 0;
float Fa = 0.0;
uint16_t temp_Fa[20] = {0};

void ktkt(){
    DEBUG_PRINT("range\n");
}

uint16_t s1_calFa(){ // measurement: mili Newtow (mN)
  accZ = logGetUint(logGetVarId("stateEstimateZ","az"));
  motor1 = logGetUint(logGetVarId("motor","m1s"));
  motor2 = logGetUint(logGetVarId("motor","m2s"));
  motor3 = logGetUint(logGetVarId("motor","m3s"));
  motor4 = logGetUint(logGetVarId("motor","m4s"));
  // uint16_t rpm = -0.0008596853063780414 * accZ * accZ * accZ * accZ + 0.19747266568055943 * accZ * accZ * accZ -16.0526867466245 * accZ * accZ + 727.3007783602438 * accZ + 244;
  r_square= motor1*motor1 + motor2*motor2 + motor3*motor3 + motor4*motor4;
  thrust = C_t_fixed / rpm2rs * rho * D*D*D*D * r_square;
  Fa = (uint16_t)(mass * accZ) - thrust*1000; // asume R=[0,0,1]
  return (uint16_t)Fa; // measurement: mili Newtow (mN)
}

// bool s1_detect(){
//   temp_Fa = s1_calFa()


// }



void appMain() {
  DEBUG_PRINT("Hello! I am the stm_gap8_cpx app\n");

  // Register a callback for CPX packets.
  // Packets sent to destination=CPX_T_STM32 and function=CPX_F_APP will arrive here
  cpxRegisterAppMessageHandler(cpxPacketCallback);

  uint8_t counter = 0;
  while(1) {
    vTaskDelay(M2T(100));

    cpxInitRoute(CPX_T_STM32, CPX_T_GAP8, CPX_F_APP, &txPacket.route);
    txPacket.data[0] = counter;
    // txPacket.data[1] = counter+1;
    // txPacket.data[2] = counter+2;
    txPacket.dataLength = 1;
    // uint16_t zrange = logGetUint(logGetVarId("range", "zrange"));
    
    cpxSendPacketBlocking(&txPacket);
    // DEBUG_PRINT("Sent packet to GAP8 (%u)\n", zrange);
    // s1_calFa();
    DEBUG_PRINT("Sent packet to GAP8 (%u)\n", s1_calFa());
    counter++;


    











  }
}

static void cpxPacketCallback(const CPXPacket_t* cpxRx) {
  DEBUG_PRINT("Got packet from GAP8 (%u)\n", cpxRx->data[0]);
}
