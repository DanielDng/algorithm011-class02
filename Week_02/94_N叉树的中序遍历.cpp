// ���õݹ����������������帨������ʵ�ֵݹ�

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
            res.push_back(root -> val); // �����һ�����ڣ���������ֵ���� res ����
            if (root -> right != NULL) {
                travel(root -> right, res);
            }
        }
    }
    
    /* ��������д����
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


// ����ջ�ı������Լ��ֶ�ά��һ��ջ
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> res;
    stack<TreeNode*> stk;
    while (root || !stk.empty()) {
        while (root) {
            stk.push(root);
            root = root -> left; // ������root�󣬼��������
        }
        root = stk.top();
        stk.pop();
        res.push_back(root -> val);
        root = root -> right; // ������root�󣬼�����ҽ��
    }
    return res;
}