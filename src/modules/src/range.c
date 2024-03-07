/**
 *    ||          ____  _ __
 * +------+      / __ )(_) /_______________ _____  ___
 * | 0xBC |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * +------+    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *  ||  ||    /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie Firmware
 *
 * Copyright (C) 2018 Bitcraze AB
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
 * ranges.c: Centralize range measurements for different directions
 *           and make them available as log
 */
#include <stdint.h>

#include "log.h"

#include "range.h"
#include "stabilizer_types.h"
#include "estimator.h"

static uint16_t ranges[RANGE_T_END] = {0,};
static uint16_t fake1 = 0;
static uint16_t fake2 = 0;
static uint16_t fake3 = 0;


void rangeSet(rangeDirection_t direction, float range_m)
{
  if (direction > (RANGE_T_END-1)) return;

  ranges[direction] = range_m * 1000;
}

float rangeGet(rangeDirection_t direction)
{
    if (direction > (RANGE_T_END-1)) return 0;

  return ranges[direction];
}


float pre_zrange = 0.0;
bool onboard = false;
int epoch = 0;
float pre_height = 0.0;
void rangeEnqueueDownRangeInEstimator(float distance, float stdDev, uint32_t timeStamp) {
  tofMeasurement_t tofData;
  tofData.timestamp = timeStamp;



  float desk_height = 0.75;
  // on board
  if ((distance - pre_zrange)<(float)(-0.600 ) ){
    onboard = true;
  }
  else if((distance - pre_zrange)>(float)(+0.600)){
    onboard = false;
  }


  float tem_height = 0.0;
  if (onboard==true){
    tem_height = distance + desk_height;
  }
  else{
    tem_height = distance;
  }
  
  fake3 = pre_height *1000;


  if(((tem_height-pre_height)>(float)0.06) || ((tem_height-pre_height)<(float)-0.06)){
    
    tofData.distance = pre_height;
    
  }
  else{

    tofData.distance = tem_height;
    pre_height = tem_height;
  }
  // tofData.distance

  epoch++;
  if (epoch % 8 == 0)
  {
    epoch = 0;
    pre_zrange = distance;}

  fake1 = tem_height * 1000;
  fake2 = tofData.distance*1000;
  tofData.stdDev = stdDev;
  estimatorEnqueueTOF(&tofData);
}

/**
 * Log group for the multi ranger and Z-ranger decks
 */
LOG_GROUP_START(range)
/**
 * @brief Distance from the front sensor to an obstacle [mm]
 */
LOG_ADD_CORE(LOG_UINT16, front, &ranges[rangeFront])

/**
 * @brief Distance from the back sensor to an obstacle [mm]
 */
LOG_ADD_CORE(LOG_UINT16, back, &ranges[rangeBack])

/**
 * @brief Distance from the top sensor to an obstacle [mm]
 */
LOG_ADD_CORE(LOG_UINT16, up, &ranges[rangeUp])

/**
 * @brief Distance from the left sensor to an obstacle [mm]
 */
LOG_ADD_CORE(LOG_UINT16, left, &ranges[rangeLeft])

/**
 * @brief Distance from the right sensor to an obstacle [mm]
 */
LOG_ADD_CORE(LOG_UINT16, right, &ranges[rangeRight])

/**
 * @brief Distance from the Z-ranger (bottom) sensor to an obstacle [mm]
 */
LOG_ADD_CORE(LOG_UINT16, zrange, &ranges[rangeDown])

/**
 * @brief Distance from the Z-ranger (bottom) sensor to an obstacle [mm]
 */
LOG_ADD_CORE(LOG_UINT16, high1, &fake2)

/**
 * @brief Distance from the Z-ranger (bottom) sensor to an obstacle [mm]
 */
LOG_ADD_CORE(LOG_UINT16, high2, &fake2)

/**
 * @brief Distance from the Z-ranger (bottom) sensor to an obstacle [mm]
 */
LOG_ADD_CORE(LOG_UINT16, high3, &fake3)


LOG_GROUP_STOP(range)
