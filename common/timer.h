#ifndef _TIMER_H_
#define _TIMER_H_

#include <stdio.h>
#include <sys/time.h>

typedef struct Timer {
    struct timeval startTime;
    struct timeval endTime;
    double         time;
} Timer;

static void startTimer(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

static void stopTimer(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

static float getElapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) * 1000000.0
                   + (timer.endTime.tv_usec - timer.startTime.tv_usec)) / 1000.0);
}

#endif