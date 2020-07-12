# 第三周总结

[TOC]

> [如何优雅的计算Fibonacci数列](https://blog.csdn.net/yjn1995/article/details/96620332)

### 递归

#### 实现特性及思维要点

> 树的面试题一般是用递归
>
> 主要原因:
>
> * 节点的定义
> * 重复性（自相似性）

递归 - 循环

通过函数体来调用自己进行循环

![CPLZK.png](https://wx1.sbimg.cn/2020/07/08/CPLZK.md.png)

![CPjIG.png](https://wx2.sbimg.cn/2020/07/08/CPjIG.md.png)

![CPqzw.png](https://wx2.sbimg.cn/2020/07/08/CPqzw.md.png)

![CPeOo.png](https://wx1.sbimg.cn/2020/07/08/CPeOo.md.png)

思维要点：

* 不要人肉进行递归（最大误区）
* 找到最近最简的方法，将其拆解成可重复解决的问题（重复子问题）
* 数学归纳法思维



### 递归的实战演练

#### [70. 爬楼梯](https://leetcode-cn.com/problems/climbing-stairs/)

1. 直接递归：（超时）

```cpp
int climbStairs(int n) {
    if (n == 1) return 1;
    if (n == 2) return 2;
    return climbStairs(n - 1) + climbStairs(n - 2);
}
```



2. 动态规矩1：$O(n)$

错误代码：

```cpp
int climbStairs(int n) {
    vector<int> dp(n+1);
    dp[1] = 1;
    dp[2] = 2;
    for (int i = 3; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

> 以上代码有bug：当 n = 1时会发生溢出。
>
> 因为 n = 1时，dp 数组的长度为 2，所以 dp[2] 发生溢出。
>
> 即 AddressSanitizer: heap-buffer-overflow

正确代码：

```cpp
int climbStairs(int n) {
    if (n == 0 || n == 1) return 1;
    vector<int> dp(n+1);
    dp[0] = 1;
    dp[1] = 1;
    for (int i = 2; i <= n; i++) {
        dp[i] = dp[i - 1] + dp[i - 2];
    }
    return dp[n];
}
```

3. 动态规划2：$O(n)$

```cpp
int climbStairs(int n) {
    if (n == 0 || n == 1) return 1;
    int a = 1, b = 1, c;
    for (int i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return c;
}
```

> 空间复杂度变为：$O(1)$



#### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

先搭建递归框架，看 2 * n 个括号有多少种组合方式：

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        generate(0, 2 * n, "");
        return res;
    }

private:
    vector<string> res; 
    void generate(int level, int maxLevel, string s) {
        // terminator
        if (level >= maxLevel) {
            res.push_back(s);
            return ;
        }

        // process current logic: left, right
        string s1 = s + "(";
        string s2 = s + ")";

        // drill down
        generate(level + 1, maxLevel, s1);
        generate(level + 1, maxLevel, s2);

        // reverse states

    }

};
```



若题目 n = 3，表示只有三个左括号、三个右括号

所以：

* 左括号添加：随时可以加，只要别超标（< 3）
* 右括号添加：必须左括号个数 > 右括号的个数

```cpp
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        generate(0, 0, n, "");
        return res;
    }

private:
    vector<string> res; 
    void generate(int left, int right, int n, string s) {
        // terminator
        if (left == n && right == n) { // 当左括号和有括号都用完时
            res.push_back(s);
            return ;
        }

        // process current logic: left, right

        // drill down
        if (left < n) generate(left + 1, right, n, s + "(");

        if (left > right) generate(left, right + 1, n, s + ")");
        
        // reverse states 清理当前层
    }
};
```

> 时间复杂度：$O(2^n)$



#### [98. 验证二叉搜索树](https://leetcode-cn.com/problems/validate-binary-search-tree/)

1. 递归1：BST -> 它的中序遍历是递增的，所以当前驱节点不小于当前节点时，就直接返回 false

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
    bool isValidBST(TreeNode* root) {
        TreeNode* prev = NULL; // 指向当前节点的前驱节点
        return validate(root, prev);
    }

    bool validate(TreeNode* node, TreeNode* &prev) {
        if (node == NULL) return true;
        if (!validate(node -> left, prev)) return false; // 递归左子树
        if (prev != NULL && prev -> val >= node -> val) return false; // 当前驱节点不小于当前节点时，返回 false
        prev = node;
        return validate(node -> right, prev); 
    }
};
```



2. 递归2：没看懂

```cpp
bool isValidBST(TreeNode* root) {
    return isValidBST(root, NULL, NULL);
}

bool isValidBST(TreeNode* root, TreeNode* minNode, TreeNode* maxNode) {
    if(!root) return true;
    if(minNode && root->val <= minNode->val || maxNode && root->val >= maxNode->val)
        return false;
    return isValidBST(root->left, minNode, root) && isValidBST(root->right, root, maxNode);
}
```



#### [104. 二叉树的最大深度](https://leetcode-cn.com/problems/maximum-depth-of-binary-tree/)

1. 深度优先搜索DFS，直接递归调用：

```cpp
int maxDepth(TreeNode* root) {    
    if (!root) return 0;
    int left = maxDepth(root -> left) + 1;
    int right = maxDepth(root -> right) + 1;
    return max(left, right);
}
```



国际友人写法：

```cpp
int maxDepth(TreeNode *root)
{
    return root == NULL ? 0 : max(maxDepth(root -> left), maxDepth(root -> right)) + 1;
}
```



2. 广度优先搜索 BFS：

```cpp
int maxDepth(TreeNode* root) {
    if (!root) {
        return 0;
    }
    int res = 0;
    queue<TreeNode *> q;
    q.push(root);
    while (!q.empty()) {
        ++res;
        for (int i = 0, n = q.size(); i < n; ++i) { // 遍历队列中的每个节点
            TreeNode *p = q.front();
            q.pop();
            if (p -> left) {
                q.push(p -> left);
            }
            if (p -> right) {
                q.push(p -> right);
            }
        }
    }
    return res;
}
```



### 分治、回溯

#### 分治（Divide and Conquer）

分治即是一种递归

* 找重复性

* 分解问题
* 组合每个子问题的结果

![C6jIe.png](https://wx2.sbimg.cn/2020/07/08/C6jIe.md.png)

![C6qzN.png](https://wx2.sbimg.cn/2020/07/08/C6qzN.md.png)

![C681o.png](https://wx2.sbimg.cn/2020/07/08/C681o.md.png)

> 跟泛型递归的不同：最后 result 需要将每个子结果组装成最终的结果，然后返回
>
> recursion terminator：问题被解决了，没有子问题需要被解决了，即到达叶子节点
>
> prepare data：处理当前逻辑，当前的大问题如何分解成小问题
>
> conquer subproblems：drill down，下探到下一层，解决更细节的子问题
>
> process and generate the final result：将子问题组合成原问题，然后返回



#### 回溯（Backtracking）

> 可以视为一种递归的情况

回溯法采用试错的思想，它尝试分步的去解决一个问题。在分步解决问题的过程中，当它通过尝试发现现有的分步答案不能得到有效的正确的解答的时候，它将取消上一步甚至是上几步的计算，再通过其它的可能的分步解答再次尝试寻找问题的答案。回溯法通常用最简单的递归方法来实现，在反复重复上述的步骤后可能出现两种情况：

* 找到一个可能存在的正确的答案
* 在尝试了所有可能的分步方法后宣告该问题没有答案

在最坏的情况下，回溯法会导致一次复杂度为指数时间的计算。



### 分治和回溯实战演练

#### [50. Pow(x, n)](https://leetcode-cn.com/problems/powx-n/)

分治

> 时间复杂度：$O(\log n)$，即递归的层数
>
> 空间复杂度：$O(\log n)$，即递归的层数。这是由于递归的函数调用会使用栈空间。

```cpp
class Solution {
public:
    double quickPow(double x, long long N) {
        if (N == 0) return 1.0;
        double y = quickPow(x, N / 2);
        return N % 2 == 0 ? y * y : y * y * x;
    }

    double myPow(double x, int n) {
        long long N = n;
        return N > 0 ? quickPow(x, N) : 1.0 / quickPow(x, -N);
    }
};
```



二进制运算：

```cpp
double myPow(double x, int n) {
    double ans = 1.0;
    long long absN = abs(n);
    while (absN > 0) {
        if (absN & 1) ans *= x; // & 二进制逻辑与运算，取最末位的值，如果当前的末位为1，则乘以相应的2的多少次方。
        x *= x;
        absN >>= 1; // absN = absN >> 1; 相当于 absN /= 2;
    }
    return n < 0 ? 1 / ans : ans;
}
```

> 时间复杂度：$O(\log n)$，即为对 n 进行二进制拆分的时间复杂度。
>
> 空间复杂度：$O(1)$ 。

> 左移运算符 `<<`：左移 n 位，相当于乘以 2 的 n 次方。
>
> 右移运算符 `>>`：右移 n 位，相当于除以 2 的 n 次方。
>
> `&` 运算通常用于二进制取位操作，例如一个数 `&1`的结果就是取二进制的最末位。



#### [78. 子集](https://leetcode-cn.com/problems/subsets/)

1. 递归：每个元素有两种选择，选或者是不选（注意最后要 remove）

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans; // 存储最后的结果
        if (nums.size() == 0) return ans;
        vector<int> list; // 存储中间结果
        dfs(ans, nums, list, 0);
        return ans;
    }

private:
    void dfs(vector<vector<int>>& ans, vector<int>& nums, vector<int>& list, int index) {
        // terminator
        if (index == nums.size()) { // 走到最末层，我们就可以得到结果了，把结果放到 ans 里面
            ans.push_back(list);
            return ; // 不要漏掉
        }
		// 每一层可选可不选
        dfs(ans, nums, list, index + 1); // not pick the number at this index

        list.push_back(nums[index]);
        dfs(ans, nums, list, index + 1); // pick the number at this index

        // reverse the current state
        list.pop_back(); // 因为 list 是作为递归函数的引用参数，不reverse的话会改变上面几层的函数
    }
    
};
```

> 或者最后不用 reverse states，而是将每个`dfs(ans, nums, list, index + 1);` 里面的参数 `list` 变为 `list` 的拷贝，如 java：`dfs(ans, nums, list.clone(), index + 1);`

> 为什么递归函数中的参数 `list` 会不断的变化：因为这里的 `list`传递的参数是引用型的参数，不管在函数的什么地方，`list`都会被改变，进而影响其他层的递归函数的执行过程，因此要在对`list` add 之后，再remove 一下，使其他层的 `list`保持不变。



或者：

把函数 `dfs` 中的参数 `vector<int>& list` 改为  `vector<int> list`，这样传的就是 `list` 的拷贝，就不再是引用了，也不需要最后 remove 一下了。

```cpp
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> ans;
        if (nums.size() == 0) return ans;
        vector<int> list;
        dfs(ans, nums, list, 0);
        return ans;
    }

private:
    void dfs(vector<vector<int>>& ans, vector<int>& nums, vector<int> list, int index) {
        // terminator
        if (index == nums.size()) {
            ans.push_back(list);
            return ;
        }

        dfs(ans, nums, list, index + 1); // not pick the number at this index

        list.push_back(nums[index]);
        dfs(ans, nums, list, index + 1); // pick the number at this index

        // reverse the current state
        // list.pop_back();
    } 
};
```



2. 迭代

Using `[1, 2, 3]` as an example, the iterative process is like:

1. Initially, one empty subset `[[]]`
2. Adding `1` to `[]`:  `[[], [1]]`;
3. Adding `2` to `[]` and `[1]`:  `[[], [1], [2], [1, 2]]`;
4. Adding `3` to `[]`, `[1]`, `[2]` and `[1, 2]`:  `[[], [1], [2], [1, 2], [3], [1, 3], [2, 3], [1, 2, 3]]`.

```cpp
vector<vector<int>> subsets(vector<int>& nums) {
    vector<vector<int>> subs = {{}};
    for (int num : nums) {
        int n = subs.size();
        for (int i = 0; i < n; ++i){
            subs.push_back(subs[i]); // 将当前遍历的元素再取出来，再 push 进去
            subs.back().push_back(num); // 对取出来的元素再往里加 nums 里面的元素
        }
    }
    return subs;
}
```



#### [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

[**牛顿迭代法** 在官方题解中的介绍](https://leetcode-cn.com/problems/sqrtx/solution/x-de-ping-fang-gen-by-leetcode-solution/)

[牛顿迭代法代码](http://www.voidcn.com/article/p-eudisdmk-zm.html)

```cpp
int mySqrt(int x) {
    if (x == 0) {
        return 0;
    }

    double C = x, x0 = x;
    while (true) {
        double xi = 0.5 * (x0 + C / x0);
        if (fabs(x0 - xi) < 1e-7) { // 相邻两次迭代的结果的差值如果小于一个极小的非负数，迭代终止
            break;
        }
        x0 = xi;
    }
    return int(x0);
}
```

> 时间复杂度：$O(\log x)$，此方法是二次收敛的，相较于二分查找更快
>
> 空间复杂度：$O(1)$



#### [169. 多数元素](https://leetcode-cn.com/problems/majority-element/)

1. hash

```cpp
int majorityElement(vector<int>& nums) {
    unordered_map<int, int> counts;
    int cnt = nums.size() / 2;
    for (int num : nums) {
        // counts[num]++;
        if (++ counts[num] > cnt) {
            return num;
        }
    }
    return -1;
}
```

> 时间复杂度：$O(n)$
>
> 空间复杂度：$O(n)$



2. 排序

先对整个数组进行排序，因为众数出现的频率大于`n/2`，所以排序后的中间位置必然是众数

```cpp
int majorityElement(vector<int>& nums) {
    sort(nums.begin(), nums.end());
    return nums[nums.size() / 2];
}
```



3. 分治

假设数 `a` 是数组 `nums` 的众数，如果我们将 `nums` 分成两部分，那么 `a` 必定是至少一部分的众数。

这样以来，我们就可以使用分治法解决这个问题：将数组分成左右两部分，分别求出左半部分的众数 `a1` 以及右半部分的众数 `a2`，随后在 `a1` 和 `a2` 中选出正确的众数。

**算法**：

我们使用经典的分治算法递归求解，直到所有的子问题都是长度为 `1` 的数组。长度为 `1` 的子数组中唯一的数显然是众数，直接返回即可。如果回溯后某区间的长度大于 `1`，我们必须将左右子区间的值合并。如果它们的众数相同，那么显然这一段区间的众数是它们相同的值。否则，我们需要比较两个众数在整个区间内出现的次数来决定该区间的众数。

```cpp
// 分治
class Solution {
private:
    int majorityElementRec(vector<int>& nums, int left, int right) {
        if (left == right) return nums[left];
        int mid = (left + right) / 2;
        int leftMajority = majorityElementRec(nums, left, mid);
        int rightMajority = majorityElementRec(nums, mid + 1, right);
        
        // 分别计算左边及右边的众数出现的次数，挑选出大于 n/2 次数的数
        if (countMajority(nums, leftMajority, left, right) > (right - left + 1) / 2)
            return leftMajority;

        if (countMajority(nums, rightMajority, left, right) > (right - left + 1) / 2)
            return rightMajority;
        return -1;
    }

    // 计算众数出现的次数
    int countMajority(vector<int>& nums, int target, int left, int right) {
        int count = 0;
        for (int i = left; i <= right; i++) {
            if (nums[i] == target) count++;
        }
        return count;
    }

public:
    int majorityElement(vector<int>& nums) {
        return majorityElementRec(nums, 0, nums.size() - 1);
    }
};
```



#### [17. 电话号码的字母组合](https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/)

1. Iteration solution

```cpp
vector<string> letterCombinations(string digits) {
    vector<string> result;
    if (digits.empty()) return {};
    static const vector<string> v = {"", "", "abc", "def", "ghi", "jkl", 
                                     "mno", "pqrs", "tuv", "wxyz"};

    result.push_back(""); // add a seed for the initial case
    for (int i = 0; i < digits.size(); ++i) {
        int num = digits[i] - '0';
        if (num < 0 || num > 9) break;
        const string candidate = v[num];
        if (candidate.empty()) continue;

        vector<string> tmp;
        for (auto can : candidate) {
            for (auto res : result) {
                tmp.push_back(res + can);
            }
        }
        result = tmp;
        // result.swap(tmp); 
        // vector::swap() swaps the reference address in the two vectors and takes O(1) whereas simple assignment takes O(n), though of course it is not necessary.
    }
    return result;
}
```



Explanation with sample input "23"

Initial state:

* result = {""}



Stage 1 for number "2":

- result has {""}
- candidate is "abc"
- generate three strings "" + "a", ""+"b", ""+"c" and put into tmp,
  tmp = {"a", "b","c"}
- result = tmp
- Now result has {"a", "b", "c"}



Stage 2 for number "3":

- result has {"a", "b", "c"}
- candidate is "def"
- generate nine strings and put into tmp,
  "a" + "d", "a"+"e", "a"+"f",
  "b" + "d", "b"+"e", "b"+"f",
  "c" + "d", "c"+"e", "c"+"f"
- so tmp has {"ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf" }
- result = tmp
- Now result has {"ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf" }

Finally, return result.



More clearer code:

```cpp
vector<string> letterCombinations(string digits) {
    vector<string> res;
    string charmap[10] = {"0", "1", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    res.push_back("");
    for (int i = 0; i < digits.size(); i++)
    {
        vector<string> tempres;
        string chars = charmap[digits[i] - '0'];
        for (int c = 0; c < chars.size();c++)
            for (int j = 0; j < res.size();j++)
                tempres.push_back(res[j]+chars[c]);
        res = tempres;
    }
    return res;
}
```



2. 递归（类比 [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)）

```cpp
class Solution {
private:
    void search(string s, string digits, int i, vector<string>& res, 
        unordered_map<char, string>& map) { // i 即 level
        // terminator
        if (i == digits.size()) { // 全部走完
            res.push_back(s);
            return ;
        }

        // process the current level
        string letters = map[digits[i]];
        for (auto l : letters) {
            // drill down
            search(s + l, digits, i + 1, res, map);
        }

        // reverse
    }
public:
    vector<string> letterCombinations(string digits) {
        if (digits.empty()) return {};
        unordered_map<char, string> map;
        map['2'] = "abc";
        map['3'] = "def";
        map['4'] = "ghi";
        map['5'] = "jkl";
        map['6'] = "mno";
        map['7'] = "pqrs";
        map['8'] = "tuv";
        map['9'] = "wxyz";
        vector<string> res;
        search("", digits, 0, res, map);
        return res;
    }
};
```



#### [51. N皇后](https://leetcode-cn.com/problems/n-queens/)

回溯

> 记得 reverse the states，因为 nQueens 也作为参数的一部分

```cpp
class Solution {
private:
    void solveNQueens(vector<vector<string>>& res, vector<string>& nQueens, int row, int n) {
        // terminator
        if (row == n) {
            res.push_back(nQueens);
            return ;
        }

        // process the current states
        for (int col = 0; col < n; ++col) {
            if (isValid(nQueens, row, col, n)) {
                nQueens[row][col] = 'Q';
                // drill down
                solveNQueens(res, nQueens, row + 1, n);
                // reverse the states
                nQueens[row][col] = '.';
            }
        }
    }

    bool isValid(vector<string>& nQueens, int row, int col, int n) { // check the queens before
        // check the column
        for (int i = 0; i < row; ++i) {
            if (nQueens[i][col] == 'Q') return false;
        }
        // check the 45 diagnol
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; --i, --j) {
            if (nQueens[i][j] == 'Q') return false;
        }
        // check the 135 diagnol
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; --i, ++j) {
            if (nQueens[i][j] == 'Q') return false;
        }
        return true;
    }

public:
    vector<vector<string>> solveNQueens(int n) {
        vector<vector<string>> res; // 三维数组，因为结果可能不止一种情况
        vector<string> nQueens(n, string(n, '.')); // 存每种解决方案的每一行
        solveNQueens(res, nQueens, 0, n);
        return res;
    }
};
```



超哥的回溯：

```python
class Solution:
    def solveNQueens(self, n: int) -> List[List[str]]:
        if n < 1: return []
        
        self.result = []
        # 之前的皇后所能够攻击的位置 (列、pie、na)
        self.cols = set()
        self.pie = set()
        self.na = set()

        self.DFS(n, 0, [])
        return self._generate_result(n)
    
    def DFS(self, n, row, cur_state):
        # recursion terminator
        if row >= n:
            self.result.append(cur_state) # 将 column 进行存储
            return

        # process the current level
        for col in range(n): # 遍历 column
            if col in self.cols or row + col in self.pie or row - col in self.na:
                # go die
                continue
            
            # update the flags
            self.cols.add(col)
            self.pie.add(row + col)
            self.na.add(row - col)

            self.DFS(n, row + 1, cur_state + [col])

            # reverse the states
            self.cols.remove(col)
            self.pie.remove(row + col)
            self.na.remove(row - col)

    def _generate_result(self, n):
        board = []
        for res in self.result:
            for i in res:
                board.append("." * i + "Q" + "." * (n - i - 1))
        return [board[i : i + n] for i in range(0, len(board), n)]
```



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



