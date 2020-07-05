// 递归实现
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        traversal(root, res);
        return res;
    }
private:
    void traversal(TreeNode* root, vector<int>& res) {
        if (!root) {
            return ; // 根节点为空时，直接返回上一级
        }
        res.push_back(root -> val);
        traversal(root -> left, res);
        traversal(root -> right, res);
    }
};

// 使用栈来模拟递归实现
vector<int> preorderTraversal(TreeNode* root) {
    vector<int> res;
    if (!root) {
        return res;
    }
    
    stack<TreeNode*> stk;
    stk.push(root); // 根结点先入栈
    while (!stk.empty()) {
        root = stk.top();
        stk.pop();
        res.push_back(root -> val); // 出栈的同时存储根结点的值
        if (root -> right) {
            stk.push(root -> right); // 先进后出，右孩子进栈
        }
        if (root -> left) {
            stk.push(root -> left); // 左孩子进栈
        }
    }
    return res;
}