#pragma once
// interesting thought: FIX capacity forever and keep track of min to get forget min behave
// interesting thought: use more memory but store branch decomposition for prefetch madness
// Priority queues with P priorities and V vals
template <typename P,typename V> class pq {

    // Data sizes for allocation purposes
    size_t psz = sizeof(P), vsz = sizeof(V);

    // Priority scalars
    P *prio;

    // Value data
    V *vals;

    // Capacity information
    int room;
    int sz;

    void swap(int i, int j) {
        /* Swap prio */ P p = prio[i]; prio[i] = prio[j]; prio[j] = p;
        /* Swap vals */ V v = vals[i]; vals[i] = vals[j]; vals[j] = v;
    }

    void sink(int i) {
        int j = i << 1;
        while (j <= sz) {
            if (j < sz && prio[j] > prio[j|1]) j++;
            if (prio[i] < prio[j]) return;
            swap(i, j);
            i = j;
            j = i << 1;
        }
    }

    void swim(int i) {
        int j = i >> 1;
        while (i > 1 && prio[j] >  prio[i]) {
            swap(j, i);
            i = j;
            j = i >> 1;
        }
    }


public:
    
    ~pq() {
      free(prio);
      free(vals);
    }

    pq(): pq(1 << 9) {}

    pq(int initialCapacity): room(initialCapacity + 1), sz(0) {
        prio = static_cast<P *>(malloc(psz * (room + 1)));
        vals = static_cast<V *>(malloc(vsz * (room + 1)));
    }

    void push(P p, V v) {
        sz++;
        prio[sz] = p;
        vals[sz] = v;
        swim(sz);
    }

    V pop() {
        swap(1, sz);
        sink(1);
        return vals[sz--];
    }

    V peek() {
        return vals[1];
    }

    P minimum() const {
        return prio[1];
    }

    void clear() {
        sz = 0;
    }

    bool empty() {
      return sz > 0;
    }

    int size() {
      return sz;
    }

    //void swap(pq other) {
    //  auto ptmp = other.prio;
    //  auto vtmp = other.vals;
    //  other.prio = prio;
    //  other.vals = vals;
    //  prio = ptmp;
    //  vals = vtmp;
    //}

};

