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


#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include "app.h"
#include "arm_math.h"
#include "arm_const_structs.h"
#include "cpx.h"
#include "cpx_internal_router.h"
#include "log.h"
#include "FreeRTOS.h"
#include "task.h"

#define DEBUG_MODULE "APP"
#include "debug.h"

int16_t accZ = 0;
uint16_t motor1 = 0;
uint16_t motor2 = 0;
uint16_t motor3 = 0;
uint16_t motor4 = 0;
uint32_t r_square = 0;

// int accZ = 0;
// int motor1 = 0;
// int motor2 = 0;
// int motor3 = 0;
// int motor4 = 0;
// int r_square = 0;
float thrust = 0;
float Fa = 0.0;

#define mass 0.036 // mass
#define g 9.81 // gravity
#define rho 1.225 // air density
#define D 0.05  // diameter of the rotor
#define rpm2rs 3600
#define C_t_fixed 0.09937873049125099
#define DATA_DIMENSION 1

// #include <stdio.h>
#include <math.h>
#include <stdlib.h>
// #include <string.h>

// #define PI 3.14159265358979323846
#define RAWDATA_SIZE 128
// 复数结构体


typedef struct {
    double data[RAWDATA_SIZE];  // 二维数组
    int front;
    int rear;
    int size; 
} CircularQueue;

typedef struct {
    int32_t data[61];  // 二维数组
    int front;
    int rear;
    int size; 
} CircularQueueCFAR;


typedef struct {
    int16_t data[10];  // 二维数组
    int front;
    int rear;
    int size; 
} CircularQueue10;

void cfar_so(CircularQueueCFAR *queue, int N, int pro_N, double PAD, double *XT, uint8_t* target_s2);
double average(double *arr, int len);


void initQueue(CircularQueue *q) {
    q->front = 0;
    q->rear = 0;
    q->size = 0;
}
void initQueueCFAR(CircularQueueCFAR *q) {
    q->front = 0;
    q->rear = 0;
    q->size = 0;
}
void initQueue10(CircularQueue10 *q) {
    q->front = 0;
    q->rear = 0;
    q->size = 0;
}


double average(double *arr, int len) {
    double sum = 0.0;
    for (int i = 0; i < len; i++) {
        sum += arr[i];
    }
    return sum / len;
}

int isFull(CircularQueue *q) {
    return q->size == RAWDATA_SIZE;
}

int isEmpty(CircularQueue *q) {
    return q->size == 0;
}

int isFullCFAR(CircularQueueCFAR *q) {
    return q->size == 61;
}

int isEmptyCFAR(CircularQueueCFAR *q) {
    return q->size == 0;
}

int isFull10(CircularQueue10 *q) {
    return q->size == 10;
}

int isEmpty10(CircularQueue10 *q) {
    return q->size == 0;
}

void enqueue(CircularQueue *queue, int16_t element) {
    if (queue->size >= RAWDATA_SIZE) {
        for (int i = 0; i < RAWDATA_SIZE - 1; i++) {
            for (int j = 0; j < DATA_DIMENSION; j++) {
                queue->data[i * DATA_DIMENSION + j] = queue->data[(i + 1) * DATA_DIMENSION + j];
            }
        }
        queue->rear = (RAWDATA_SIZE - 1) * DATA_DIMENSION; // Update rear index

    } else {
        queue->size++;
    }
    // Copy new element to the rear position
    for (int i = 0; i < DATA_DIMENSION; i++) {
        queue->data[queue->rear + i] = element;
    }
    queue->rear = (queue->rear + DATA_DIMENSION) % (RAWDATA_SIZE * DATA_DIMENSION);


}

















void enqueueCFAR(CircularQueueCFAR *q, int16_t data) {
    if (isFullCFAR(q)) {
        // 如果队列已满，弹出队列头部的数据
        q->front = (q->front + 1) % 61;
        q->size--;
    }
  
    q->data[q->rear] = data;
    
    q->rear = (q->rear + 1) % 61;
    q->size++;
}

void enqueue10(CircularQueue10 *q, int16_t data) {
    if (isFull10(q)) {
        // 如果队列已满，弹出队列头部的数据
        q->front = (q->front + 1) % 10;
        q->size--;
    }
  
    q->data[q->rear] = data;
    
    q->rear = (q->rear + 1) % 10;
    q->size++;
}

int STFTandextraction(CircularQueue* queue, int segmentLength) {
    float32_t* signal = (float32_t*)malloc(2*segmentLength*sizeof(int));
    int index = queue->front;
    int i=0;
    while (i<2*segmentLength) {
        // printf("%d ", queue->rear);
        index = (index + 1) % segmentLength;
        signal[i] = (float32_t)queue->data[index];
        signal[i+1] = 0;
        i++;
        i++;
    }
    
        
    for (int j = 0; j < segmentLength; ++j) {
            signal[2*j] *= 0.54f - 0.46f * arm_cos_f32(2 * PI * j / (segmentLength - 1));
            signal[2*j+1] = 0;
            // DEBUG_PRINT("raw data (%lu) %lu\n", (uint32_t)signal[2*j],(uint32_t)signal[2*j+1]);
        }


    arm_cfft_f32(&arm_cfft_sR_f32_len128,signal,0,1);
    // DEBUG_PRINT("entry (%u) %u\n", (uint16_t)signal[2],(uint16_t)signal[3]);
    // // arm_cmplx_mag_f32(signal,fft_outputbuf,segmentLength);
    // // 执行快速傅里叶变换（FFT）
    // // FFT(spectrum, segmentLength);

    // // 将傅里叶变换结果的幅度存储到STFT结果数组中
    int mag = 0;
    for (int j = 6; j < 8; ++j) {
        mag += signal[2*j]*signal[2*j]+signal[2*j+1]*signal[2*j+1];
    }

    free(signal);
  
    return mag;

}




void cfar_so(CircularQueueCFAR *queue, int N, int pro_N, double PAD, double *XT, uint8_t* target_s2) {
    
    double alpha = N * (pow(PAD, -1.0 / N) - 1);
    double left_N[N/2];

    int index = queue->front-1;
    int i=0;
    while (i<50) {
        // printf("%d ", queue->rear);
        index = (index + 1) % 61;
        left_N[i] = queue->data[index];
        i++;
    }



    double Z = average(left_N, 50);
    *XT = Z * alpha;
    // printf("###%d\n",queue->data[queue->rear-1]);
    if (queue->data[queue->rear-1] > Z * alpha) {
        
        *target_s2 = 1;
       
    }
    
}






int16_t s1_calFa(double accZ, double motor1, double motor2,double motor3,double motor4){ // measurement: mili Newtow (mN)
  
//   int m1_rpm = -0.0008596853063780414 * accZ * accZ * accZ * accZ + 0.19747266568055943 * accZ * accZ * accZ -16.0526867466245 * accZ * accZ + 727.3007783602438 * accZ + 244;
    r_square= (motor1*motor1 + motor2*motor2 + motor3*motor3 + motor4*motor4)*0.18*0.24;
    thrust = C_t_fixed / rpm2rs * rho * D*D*D*D * 4 * r_square;
    Fa = (int16_t)(mass * accZ) - thrust*1000; // asume R=[0,0,1]
    return (int16_t)Fa; // measurement: mili Newtow (mN)
}



void appMain() {
    DEBUG_PRINT("Hello! I am the stm_gap8_cpx app\n");


    // print through cpx
    DEBUG_PRINT("Hello! I am the stm_gap8_cpx app\n");
    // Register a callback for CPX packets.
    // Packets sent to destination=CPX_T_STM32 and function=CPX_F_APP will arrive here
    // cpxRegisterAppMessageHandler(cpxPacketCallback);
    
    // 输出fusion_results
    CircularQueueCFAR* fusion_dataQueue=(CircularQueueCFAR*)malloc(sizeof(CircularQueueCFAR));
    initQueueCFAR(fusion_dataQueue);
    CircularQueue10* Fa_dataQueue = (CircularQueue10*)malloc(sizeof(CircularQueue10));
    initQueue10(Fa_dataQueue);

    // 输入信号
    CircularQueue* accZ_dataQueue = (CircularQueue*)malloc(sizeof(CircularQueue));
    CircularQueue* gyroY_dataQueue = (CircularQueue*)malloc(sizeof(CircularQueue));
    CircularQueue* motor2_1_dataQueue = (CircularQueue*)malloc((sizeof(CircularQueue)));
    CircularQueue* accX_dataQueue = (CircularQueue*)malloc(sizeof(CircularQueue));
    CircularQueue* motor3_1_dataQueue = (CircularQueue*)malloc(sizeof(CircularQueue));
    initQueue(accZ_dataQueue);
    initQueue(gyroY_dataQueue);
    initQueue(motor2_1_dataQueue);
    initQueue(accX_dataQueue);
    initQueue(motor3_1_dataQueue);
    DEBUG_PRINT("initial sucess\n");

    while(1){
        // !!! Replace following rows with LogGet() function



        vTaskDelay(M2T(10));
        int16_t accZ = logGetUint(logGetVarId("stateEstimateZ","az"));
        // int16_t accX = logGetUint(logGetVarId("stateEstimateZ","ax"));
        // int16_t gyroY = logGetUint(logGetVarId("gyro","yRaw"));
        int16_t motor1 = logGetUint(logGetVarId("motor","m1s"));
        int16_t motor2 = logGetUint(logGetVarId("motor","m2s"));
        int16_t motor3 = logGetUint(logGetVarId("motor","m3s"));
        int16_t motor4 = logGetUint(logGetVarId("motor","m4s"));
        



        // enqueue(accZ_dataQueue, accZ);
        // enqueue(gyroY_dataQueue, gyroY);
        // enqueue(motor2_1_dataQueue, motor2 - motor1);
        // enqueue(accX_dataQueue, accX);
        // enqueue(motor3_1_dataQueue, motor3 - motor1);



        // int16_t accZ = logGetUint(logGetVarId("stateEstimateZ","az"));
        // enqueue(accZ_dataQueue, (int16_t)12);
        // enqueue(gyroY_dataQueue, (int16_t)12);
        // enqueue(motor2_1_dataQueue, (int16_t)12);
        // enqueue(accX_dataQueue, 12);
        // enqueue(motor3_1_dataQueue, 12);


        // DEBUG_PRINT("is entry (%u)\n", (uint8_t)accZ);






        // int mag1 = STFTandextraction(accZ_dataQueue, RAWDATA_SIZE);
        // int mag2 = STFTandextraction(gyroY_dataQueue, RAWDATA_SIZE);
        // int mag3 = STFTandextraction(motor2_1_dataQueue, RAWDATA_SIZE);
        // int mag4 = STFTandextraction(accX_dataQueue, RAWDATA_SIZE);
        // int mag5 = STFTandextraction(motor3_1_dataQueue, RAWDATA_SIZE);
        // int fusion_result = mag1*mag2*mag3*mag4*mag5;

        // // DEBUG_PRINT("1 entry (%u)\n", (uint16_t)fusion_result);



        // // CFAR detection  Part
        // enqueueCFAR(fusion_dataQueue, fusion_result);
        // int N = 100;
        // int pro_N = 20;
        // double PAD = 1e-4;
        // double XT = 0.0;
        uint8_t target_s1 = 0;
        // uint8_t* target_s2 = 0;
        
        // cfar_so(fusion_dataQueue, N, pro_N, PAD, &XT, target_s2);

       
        
        int Fa_raw = s1_calFa(accZ, motor1/65535,motor2/65535, motor3/65535, motor4/65535);
        
        enqueue10(Fa_dataQueue,Fa_raw);
        int Fa_filtered = average((double*)Fa_dataQueue->data, 10);
        
        if(Fa_filtered>30){
            target_s1 = 1;
        }

        // uint8_t is_entry = target_s1 * (*target_s2);
        
        
        DEBUG_PRINT("is entry (%u)\n", (uint8_t)target_s1);

    }

    


}
