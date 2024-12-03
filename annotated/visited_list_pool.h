#pragma once

#include <mutex>
#include <string.h>
#include <deque>

namespace hnswlib {
typedef unsigned short int vl_type;

// What is this? just allocated memory?
// How does he use it in the alg impl?
class VisitedList {
 public:
    vl_type curV;
    vl_type *mass;
    unsigned int numelements;

    VisitedList(int numelements1) {
        curV = -1;
        numelements = numelements1;
        mass = new vl_type[numelements];
    }

    void reset() {
        curV++;
        if (curV == 0) {
            memset(mass, 0, sizeof(vl_type) * numelements);
            curV++;
        }
    }

    ~VisitedList() { delete[] mass; }
};
///////////////////////////////////////////////////////////
//
// Class for multi-threaded pool-management of VisitedLists
//
/////////////////////////////////////////////////////////

// Smart, create visitedlists to avoid de-and-reallocating memory
// over and over, however am confused about VisitedList impl!!
class VisitedListPool {

    // Deque: O(1) insert/remove at ends
    // Probably linked lists beneath for gcc?
    // https://en.cppreference.com/w/cpp/container/deque

    // But it doesn't really matter here, this is just
    // to manage 
    std::deque<VisitedList *> pool;

    // Mutex / Lock / Semaph
    std::mutex poolguard;


    int numelements;

 public:
    VisitedListPool(int initmaxpools, int numelements1) {
        numelements = numelements1;
        for (int i = 0; i < initmaxpools; i++)
            pool.push_front(new VisitedList(numelements));
    }

    VisitedList *getFreeVisitedList() {
        VisitedList *rez;
        { // Trivially lock the mutex while grabbing the list
            std::unique_lock <std::mutex> lock(poolguard);
            if (pool.size() > 0) {
                rez = pool.front();
                pool.pop_front();
            } else {
                rez = new VisitedList(numelements);
            }
        }
        rez->reset();
        return rez;
    }

    void releaseVisitedList(VisitedList *vl) {
        std::unique_lock <std::mutex> lock(poolguard);
        pool.push_front(vl);
    }

    ~VisitedListPool() {
        while (pool.size()) {
            VisitedList *rez = pool.front();
            pool.pop_front();
            delete rez;
        }
    }
};
}  // namespace hnswlib
