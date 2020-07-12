### 第三周作业

#### [236. 二叉树的最近公共祖先](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/)

递归

抛开题目，要实现的函数应该有三个功能：给定两个节点 `p` 和 `q`

1. 如果 `p` 和 `q` 都存在，则返回它们的公共祖先；
2. 如果只存在一个，则返回存在的一个；
3. 如果 `p` 和 `q` 都不存在，则返回 NULL

本题说给定的两个节点都存在，那自然还是能用上面的函数来解决。

具体思路：

（1） 如果当前结点 `root` 等于 NULL，则直接返回 NULL
（2） 如果 `root` 等于 `p` 或者 `q` ，那这棵树一定返回 `p` 或者 `q`
（3） 然后递归左右子树，因为是递归，使用函数后可认为左右子树已经算出结果，用 `left` 和 `right` 表示
（4） 此时若`left`为空，那最终结果只要看 `right`；若 `right` 为空，那最终结果只要看 `left`
（5） 如果 `left` 和 `right` 都非空，因为只给了 `p` 和 `q` 两个结点，都非空，说明一边一个，因此 `root` 是他们的最近公共祖先
（6） 如果 `left` 和 `right` 都为空，则返回空（其实已经包含在前面的情况中了）

> 时间复杂度是 $O(n)$：**每个结点最多遍历一次** 或用 主定理
>
> 空间复杂度是 $O(n)$：递归需要系统栈空间



```cpp
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q) return root;
        TreeNode* left = lowestCommonAncestor(root -> left, p, q);
        TreeNode* right = lowestCommonAncestor(root -> right, p, q);
        if (!left) return right; // left 为空，只需看 right
        if (!right) return left; // right 为空，只需看 left
        return root;
    }
};
```

-----

[使用 Notion 进行刻意练习提醒](https://leetcode-cn.com/problems/lowest-common-ancestor-of-a-binary-tree/solution/er-cha-shu-de-zui-jin-gong-gong-zu-xian-by-leetc-2/397340)

[入门视频](https://www.bilibili.com/video/BV1gQ4y1K76r )

[如何搭建的自己的home page](https://www.bilibili.com/video/BV1Zb411H7xC)



#### [105. 从前序与中序遍历序列构造二叉树](https://leetcode-cn.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/)

1. 递归

对于任意一颗树而言，前序遍历的形式总是

```
[ 根节点, [左子树的前序遍历结果], [右子树的前序遍历结果] ]
```

即根节点总是前序遍历中的第一个节点。

而中序遍历的形式总是

```
[ [左子树的中序遍历结果], 根节点, [右子树的中序遍历结果] ]
```

只要我们在中序遍历中**定位到根节点**，那么我们就可以分别知道左子树和右子树中的**节点数目**。由于同一颗**子树**的前序遍历和中序遍历的**长度**显然是**相同**的，因此我们就可以对应到前序遍历的结果中，对上述形式中的所有左右括号进行定位。

这样一来，我们就知道了左子树的前序遍历和中序遍历结果，以及右子树的前序遍历和中序遍历结果，我们就可以递归地构造出左子树和右子树，再将这两颗子树接到根节点的左右位置。

**细节**

**在中序遍历中对根节点进行定位时**，一种简单的方法是直接扫描整个中序遍历的结果并找出根节点，但这样做的时间复杂度较高。我们可以考虑使用哈希映射（HashMap）来帮助我们快速地定位根节点。对于哈希映射中的每个键值对，键表示一个元素（节点的值），值表示其在中序遍历中的出现位置。在构造二叉树的过程之前，我们可以对中序遍历的列表进行一遍扫描，就可以构造出这个哈希映射。在此后构造二叉树的过程中，我们就只需要 $O(1)$的时间对根节点进行定位了。

```cpp
class Solution {
private:
    unordered_map<int, int> index;

public:
    TreeNode* myBuildTree(const vector<int>& preorder, 
                          const vector<int>& inorder, 
                          int preorder_left, 
                          int preorder_right, 
                          int inorder_left, 
                          int inorder_right) {
        
        if (preorder_left > preorder_right) {
            return nullptr;
        }
        
        // 前序遍历中的第一个节点就是根节点
        int preorder_root = preorder_left;
        // 在中序遍历中定位根节点
        int inorder_root = index[preorder[preorder_root]];
        
        // 先把根节点建立出来
        TreeNode* root = new TreeNode(preorder[preorder_root]);
        // 得到左子树中的节点数目
        int size_left_subtree = inorder_root - inorder_left;
        // 递归地构造左子树，并连接到根节点
        // 先序遍历中「从 左边界+1 开始的 size_left_subtree」个元素就对应了中序遍历中「从 左边界 开始到 根节点定位-1」的元素
        root->left = myBuildTree(preorder, 
                                 inorder, 
                                 preorder_left + 1,
                                 preorder_left + size_left_subtree, 
                                 inorder_left, 
                                 inorder_root - 1);
        // 递归地构造右子树，并连接到根节点
        // 先序遍历中「从 左边界+1+左子树节点数目 开始到 右边界」的元素就对应了中序遍历中「从 根节点定位+1 到 右边界」的元素
        root->right = myBuildTree(preorder, 
                                  inorder, 
                                  preorder_left + size_left_subtree + 1,
                                  preorder_right, 
                                  inorder_root + 1, 
                                  inorder_right);
        return root;
    }

    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        int n = preorder.size();
        // 构造哈希映射，帮助我们快速定位根节点
        for (int i = 0; i < n; ++i) {
            index[inorder[i]] = i;
        }
        return myBuildTree(preorder, inorder, 0, n - 1, 0, n - 1);
    }
};

```



#### [77. 组合](https://leetcode-cn.com/problems/combinations/)

回溯算法 + 剪枝，[Leetcode 题解](https://leetcode-cn.com/problems/combinations/solution/hui-su-suan-fa-jian-zhi-python-dai-ma-java-dai-ma-/)

```cpp
class Solution {
private:
    vector<vector<int>> res;
    void dfs(int n, int k, int start, vector<int>& path) {
        // terminator
        if (path.size() == k) {
            res.push_back(path);
            return ;
        }
        // process
        // for (int i = start; i <= n; ++i) { 未剪枝
        for (int i = start; i <= n - (k - path.size()) + 1; ++i) { // 剪枝后
            path.push_back(i);
            // drill down
            dfs(n, k, i + 1, path);
            // reverse states
            path.pop_back();
        }
    }

public:
    vector<vector<int>> combine(int n, int k) {
        if (n <= 0 || k <= 0 || k > n) return {};
        vector<int> path;
        dfs(n, k, 1, path);
        return res;
    }
};
```

剪枝举例：

如果 $n = 6$ ，$k = 4$，
`pre.size() == 1` 的时候，接下来要选择 `3` 个元素， $i$ 最大的值是 `4`，最后一个被选的是 `[4,5,6]`；
`pre.size() == 2` 的时候，接下来要选择 `2` 个元素， $i$ 最大的值是 `5`，最后一个被选的是 `[5,6]`；
`pre.size() == 3` 的时候，接下来要选择 `1` 个元素， $i$ 最大的值是 `6`，最后一个被选的是 `[6]`；

可以发现 `max(i)` 与 接下来要选择的元素貌似有一点关系，很容易知道：
`max(i) + 接下来要选择的元素个数 - 1 = n`，其中， 接下来要选择的元素个数 `= k - pre.size()`，整理得到：

```cpp
max(i) = n - (k - pre.size()) + 1
```

所以，我们的剪枝过程就是：把 `i <= n` 改成 `i <= n - (k - pre.size()) + 1` 



#### [46. 全排列](https://leetcode-cn.com/problems/permutations/)

[回溯](https://leetcode-cn.com/problems/permutations/solution/hui-su-suan-fa-xiang-jie-by-labuladong-2/)

```cpp
class Solution {
private:
    vector<vector<int>> res;
    void dfs(vector<int>& nums, vector<int>& track) {
        // terminator
        if (track.size() == nums.size()) { // track 放满了
            res.push_back(track);
            return ;
        }

        // process
        for (int i = 0; i < nums.size(); ++i) { // i < nums.size(); 对所有的元素进行遍历
            vector<int>::iterator iter = find(track.begin(), track.end(), nums[i]); // 看 track 中是否已经存在 nums[i] 元素，存在则返回下标，否则返回最后一个元素的后一个位置的下标
            if (iter != track.end()) continue; // track 中已经存在了，进入下一次循环，以下不执行
            // drill down
            track.push_back(nums[i]);
            dfs(nums, track);

            // reverse states
            track.pop_back();
        }
    }
public:
    vector<vector<int>> permute(vector<int>& nums) {
        int n = nums.size();
        if (n == 0) return {};
        // 记录路径
        vector<int> track;
        dfs(nums, track);
        return res;
    }
};
```



#### [47. 全排列 II](https://leetcode-cn.com/problems/permutations-ii/)

