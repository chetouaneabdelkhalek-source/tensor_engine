#include "traversals.hpp"
#include <iostream>
#include <stack>
#include <vector>

void stack_inOrderTraversal(node* root)
{
    std::stack<node *> stack;
    std::vector<node *> result;

    while (root != nullptr || !stack.empty())
    {
        if (root != nullptr)
        {
            stack.push(root);
            root = root->left;
        }
        else
        {
            root = stack.top();
            stack.pop();
            result.push_back(root); 
            root = root->right;
        }
    }

    for (node *n : result)
    {
        std::cout << n->data << " ";
    }
}

void stack_postOrderTraversal(node* root)
{
    if (root == nullptr)
        return;
    std::stack<node *> stack_1;
    std::stack<node *> stack_2;
    std::vector<node *> result;
    stack_1.push(root);
    while (!stack_1.empty())
    {
        root = stack_1.top();
        stack_2.push(root);
        stack_1.pop();

        if (root->left)
            stack_1.push(root->left);
        if (root->right)
            stack_1.push(root->right);
    }
    while (!stack_2.empty())
    {
        root = stack_2.top();
        result.push_back(root);
        stack_2.pop();
    }
    for (node *n : result)
    {
        std::cout << n->data << " ";
    }
}

void stack_preOrderTraversal(node* root)
{
    if (root == nullptr)
        return;
    std::stack<node *> stack;
    std::vector<node *> result;

    while (root != nullptr || !stack.empty())
    {
        if (root)
        {
            stack.push(root);
            result.push_back(root);
            root = root->left;
        }
        else
        {
            root = stack.top()->right;
            stack.pop();
        }
    }

    for (node *n : result)
    {
        std::cout << n->data << " ";
    }
}