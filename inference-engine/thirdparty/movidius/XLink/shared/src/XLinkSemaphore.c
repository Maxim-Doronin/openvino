// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// ------------------------------------
// XLink semaphores implementation. Begin.
// ------------------------------------

#include "XLinkSemaphore.h"

static pthread_mutex_t ref_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t ref_cond = PTHREAD_COND_INITIALIZER;

int XLink_sem_init(XLink_sem_t* sem, int pshared, unsigned int value)
{
    int rc = sem_init(&sem->sem, pshared, value);
    if (rc) {
        return rc;
    }

    sem->refs = 0;
    return 0;
}

int XLink_sem_destroy(XLink_sem_t* sem)
{
    int rc = 0;

    rc = pthread_mutex_lock(&ref_mutex);
    if (rc) {
        return rc;
    }

    while (sem->refs > 0) {
        rc = pthread_cond_wait(&ref_cond, &ref_mutex);
        if (rc) {
            return rc;
        }
    }
    rc = sem_destroy(&sem->sem);
    if (rc) {
        pthread_mutex_unlock(&ref_mutex);
        return rc;
    }

    return pthread_mutex_unlock(&ref_mutex);
}

int XLink_sem_post(XLink_sem_t* sem)
{
    return sem_post(&sem->sem);
}

int XLink_sem_wait(XLink_sem_t* sem)
{
    int rc = 0;

    rc = pthread_mutex_lock(&ref_mutex);
    if (rc) {
        return rc;
    }
    sem->refs++;
    rc = pthread_mutex_unlock(&ref_mutex);
    if (rc) {
        return rc;
    }

    int rc_wait = sem_wait(&sem->sem);

    pthread_mutex_lock(&ref_mutex);
    sem->refs--;
    pthread_cond_signal(&ref_cond);
    pthread_mutex_unlock(&ref_mutex);

    if (rc_wait) return rc_wait;
    return 0;
}

int XLink_sem_timedwait(XLink_sem_t* sem, const struct timespec *abstime)
{
    pthread_mutex_lock(&ref_mutex);
    sem->refs++;
    pthread_mutex_unlock(&ref_mutex);

    int rc_wait = sem_timedwait(&sem->sem, abstime);

    pthread_mutex_lock(&ref_mutex);
    sem->refs--;
    pthread_cond_signal(&ref_cond);
    pthread_mutex_unlock(&ref_mutex);

    if (rc_wait) return rc_wait;
    return 0;
}

// ------------------------------------
// XLink semaphores implementation.
// ------------------------------------
