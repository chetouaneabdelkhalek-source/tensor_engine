#ifndef ARENA_HPP
#define ARENA_HPP

#include <vector>

struct node
{
    int data;
    node *left = nullptr;
    node *right = nullptr;
};

class Arena
{
private:
    std::vector<node> pool;

public:
    Arena(int capacity = 100000)
    {
        pool.reserve(capacity);
    }
    node *allocate(int val)
    {
        pool.push_back(node{val});
        return &pool.back();
    }
};

#endif