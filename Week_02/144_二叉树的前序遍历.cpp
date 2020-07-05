// �ݹ�ʵ��
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
            return ; // ���ڵ�Ϊ��ʱ��ֱ�ӷ�����һ��
        }
        res.push_back(root -> val);
        traversal(root -> left, res);
        traversal(root -> right, res);
    }
};

// ʹ��ջ��ģ��ݹ�ʵ��
vector<int> preorderTraversal(TreeNode* root) {
    vector<int> res;
    if (!root) {
        return res;
    }
    
    stack<TreeNode*> stk;
    stk.push(root); // ���������ջ
    while (!stk.empty()) {
        root = stk.top();
        stk.pop();
        res.push_back(root -> val); // ��ջ��ͬʱ�洢������ֵ
        if (root -> right) {
            stk.push(root -> right); // �Ƚ�������Һ��ӽ�ջ
        }
        if (root -> left) {
            stk.push(root -> left); // ���ӽ�ջ
        }
    }
    return res;
}