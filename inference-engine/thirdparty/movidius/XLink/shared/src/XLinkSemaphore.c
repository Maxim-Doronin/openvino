// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "XLinkSemaphore.h"
#include "XLinkErrorUtils.h"
#include "XLinkLog.h"

#include <errno.h>

#ifndef XLINK_SEM_WARN_AND_RET_ERR_IF
#define XLINK_SEM_WARN_AND_RET_ERR_IF(condition, err) do { \
        if ((condition)) { \
            mvLog(MVLOG_WARN, "Condition failed: %s", #condition);\
            return (err); \
        } \
    } while(0)
#endif  // XLINK_SEM_WARN_AND_RET_ERR_IF

int XLink_sem_init(XLink_sem_t* sem, int pshared, unsigned int value)
{
    XLINK_RET_ERR_IF(sem == NULL, EINVAL);

    XLINK_RET_IF_FAIL(sem_init(&sem->psem, pshared, value));
    atomic_init(&sem->refs, 0);

    return 0;
}

int XLink_sem_destroy(XLink_sem_t* sem)
{
    XLINK_RET_ERR_IF(sem == NULL, EINVAL);
    XLINK_SEM_WARN_AND_RET_ERR_IF(atomic_load(&sem->refs) < 0, EINVAL);

    int expectedRefs = 0;
    while (!atomic_compare_exchange_weak(&sem->refs, &expectedRefs, -1)) {
        expectedRefs = 0;
    }

    return sem_destroy(&sem->psem);
}

int XLink_sem_post(XLink_sem_t* sem)
{
    XLINK_RET_ERR_IF(sem == NULL, EINVAL);
    XLINK_SEM_WARN_AND_RET_ERR_IF(atomic_load(&sem->refs) < 0, EINVAL);

    return sem_post(&sem->psem);
}

int XLink_sem_wait(XLink_sem_t* sem)
{
    XLINK_RET_ERR_IF(sem == NULL, EINVAL);
    XLINK_SEM_WARN_AND_RET_ERR_IF(atomic_load(&sem->refs) < 0, EINVAL);

    sem->refs++;
    int ret = sem_wait(&sem->psem);
    sem->refs--;

    return ret;
}

int XLink_sem_timedwait(XLink_sem_t* sem, const struct timespec* abstime)
{
    XLINK_RET_ERR_IF(sem == NULL, EINVAL);
    XLINK_SEM_WARN_AND_RET_ERR_IF(atomic_load(&sem->refs) < 0, EINVAL);

    sem->refs++;
    int ret = sem_timedwait(&sem->psem, abstime);
    sem->refs--;

    return ret;
}

int XLink_sem_set_refs(XLink_sem_t* sem, int refs)
{
    XLINK_RET_ERR_IF(sem == NULL, EINVAL);
    XLINK_SEM_WARN_AND_RET_ERR_IF(atomic_load(&sem->refs) < 0, EINVAL);

    atomic_store(&sem->refs, refs);

    return 0;
}

int XLink_sem_get_refs(XLink_sem_t* sem, int *sval)
{
    XLINK_RET_ERR_IF(sem == NULL, EINVAL);

    *sval = atomic_load(&sem->refs);

    return 0;
}
