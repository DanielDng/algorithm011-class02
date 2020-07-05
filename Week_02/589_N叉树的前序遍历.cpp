// 递归实现
/*
// Definition for a Node.
class Node {
public:
    int val;
    vector<Node*> children;

    Node() {}

    Node(int _val) {
        val = _val;
    }

    Node(int _val, vector<Node*> _children) {
        val = _val;
        children = _children;
    }
};
*/

class Solution {
public:
    vector<int> preorder(Node* root) {
        vector<int> res;
        traversal(root, res);
        return res;
    }
private:
    void traversal(Node* root, vector<int>& res) {
        if (!root) {
            return ;
        }

        res.push_back(root -> val);
        for (Node* child : root -> children) {
            traversal(child, res);
        }
    }
};

// 使用栈来模拟递归实现
vector<int> preorder(Node* root) {
    vector<int> res;
    if (!root) {
        return res;
    }

    stack<Node*> stk;
    stk.push(root); // 根结点先入栈
    while (!stk.empty()) {
        root = stk.top();
        stk.pop();
        res.push_back(root -> val); // 出栈的同时存储根结点的值
        for (int i = root -> children.size() - 1; i >= 0; --i) {
            stk.push(root -> children[i]);
        }
    }
    return res;
}