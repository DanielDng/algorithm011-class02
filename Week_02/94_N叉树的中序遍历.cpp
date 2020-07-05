// 采用递归进行中序遍历，定义辅助函数实现递归

class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) { // left-root-right
        vector<int> res;
        travel(root, res);
        return res;
    }
private:
    void travel(TreeNode* root, vector<int>& res) {
        if (root != NULL) {
            if (root -> left != NULL) {
                travel(root -> left, res);
            }
            res.push_back(root -> val); // 根结点一定存在，将根结点的值加入 res 容器
            if (root -> right != NULL) {
                travel(root -> right, res);
            }
        }
    }
    
    /* 国际友人写法：
    void travel(TreeNode* root, vector<int>& res) {
    	if (!root) {
            return;
        }
        travel(root -> left, res);
        res.push_back(root -> val);
        travel(root -> right, res);
    }
    */
};


// 基于栈的遍历，自己手动维护一个栈
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> res;
    stack<TreeNode*> stk;
    while (root || !stk.empty()) {
        while (root) {
            stk.push(root);
            root = root -> left; // 处理完root后，检查其左结点
        }
        root = stk.top();
        stk.pop();
        res.push_back(root -> val);
        root = root -> right; // 处理完root后，检查其右结点
    }
    return res;
}