#ifndef ARENA_HPP
#define ARENA_HPP

#include <cstdint> 
#include <vector>

enum class Kind : uint8_t {LEAF , BINARY , UNARY}; 
enum class Op : uint8_t{ADD, SUB, MUL, DIV, NEG,POW}; 
struct node
{
    int data;
    node *left = nullptr;
    node *right = nullptr;
};

struct TrieNode
{
    bool is_end = false;
    uint8_t child_count = 0;
    TrieNode *children[26] = {nullptr};
};
struct ParserNode{
   
    double data ; 
    int left=-1 ; 
    int right=-1 ; 
     Kind kind ; 
    Op op ; 
};
class Arena
{
private:
    std::vector<node> tree_pool;
    std::vector<TrieNode> trie_pool;
    std::vector<ParserNode> parser_pool;

public:
   // NOTE:
    // This arena assumes the reserved capacity is never exceeded.
    // If more objects are allocated than the reserved capacity,
    // std::vector may reallocate, invalidating all previously
    // returned pointers and causing bugs.
    // To support arbitrary growth, implement capacity management

    Arena(){
    } 

    void reserve_tree(int capacity = 100000)
    {
        tree_pool.reserve(capacity);
       
    }
    void reserve_trie(int capacity = 100000)
    {
        trie_pool.reserve(capacity);
       
    }

    void reserve_parser(int capacity = 100000){
          parser_pool.reserve(capacity);
    }
    node *allocate(int val)
    {
        tree_pool.push_back(node{val});
        return &tree_pool.back();
    }
    TrieNode* allocate_trie()
    {
       
        trie_pool.push_back(TrieNode{});
        return &trie_pool.back();
    }
};

#endif