/*
 * disjointSet.h
 *
 *  Created on: Jan 8, 2016
 *      Author: MOP8KOR
 */

#ifndef DISJOINTSET_H_
#define DISJOINTSET_H_

// disjoint-set forests using union-by-rank and path compression (sort of).

typedef struct {
  int rank;
  int p;
  int size;
} uni_elt;

class universe {
public:
  universe(int elements);
  ~universe();
  int find(int x);
  void join(int x, int y);
  int size(int x) const { return elts[x].size; }
  int num_sets() const { return num; }

private:
  uni_elt *elts;
  int num;
};

#endif /* DISJOINTSET_H_ */
