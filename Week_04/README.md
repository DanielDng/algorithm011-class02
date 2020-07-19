# 第四周总结

[TOC]

### 遍历搜索

在树/图/状态集中寻找特定节点

* 每个节点访问一次且仅访问一次
* 对于节点的访问顺序不限
  * 深度优先 - DFS：depth first search
  * 广度优先 - BFS：breadth first search
  * 优先级优先 - 启发式搜索：如推荐算法等



C++中**二叉树**结构的定义：

```cpp
struct TreeNode{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x): val(x), left(NULL), right(NULL) {}
}
```



Python中**二叉树**结构的定义：

```python
class TreeNode:
	def __init__(self, val):
        self.val = val
        self.left, self.right = None, None
```





#### 深度优先 - DFS

二叉树示例代码：

```python
def dfs(node):
    if node in visited:
        # already visited
        return
    
    visited.add(node)
    
    # process current node
    # ... # logic here
    dfs(node.left)
    dfs(node.right)
```



多叉树示例代码：

```python
visited = set()
def dfs(node, visited):
    visited.add(node)
    # process current node
    # ...
    
    for next_node in node.children():
        if not next_node in visited:
            dfs(next_node, visited)
```



二叉树递归**模板**的写法：

```python
visited = set()
def dfs(node, visited):
    # terminator
    if node in visited:
        # already visited
    	return
    
    visited.add(node)
    
    # process current node
    # ...
    
    for next_node in node.children():
        if not next_node in visited: # 严谨起见，再判断一次
            dfs(next_node, visited)
```



非递归的写法：手动维护一个**栈**来模拟递归

```python
def DFS(self, tree):
    if tree.root is None:
        return []
    
    visited, stack = [], [tree.root]
    
    while stack:
        node = stack.pop()
        visited.add(node)
        
        process(node)
        nodes = generete_related_nodes(node)
        stack.push(nodes)
    
    # other processing work
    # ...
```



#### 广度优先 - BFS

使用**队列**来表示：

```python
def BFS(graph, start, end):
    queue = []
    queue.append([start])
    # visited.add(start)
    
    while queue:
        node = queue.pop()
        visited.add(node)
        
        process(node)
        nodes = generate_related_nodes(node)
        queue.push(nodes)
    
    # other processing work
    # ...
```



### DFS & BFS 实战

#### [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/)

> 与 [429. N叉树的层序遍历](https://leetcode-cn.com/problems/n-ary-tree-level-order-traversal/) 题目和代码基本相同

使用队列 `queue` 进行 BFS：

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
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (!root) return res;
        queue<TreeNode*> q;
        q.push(root);
        while (!q.empty()) {
            int n = q.size(); // Store the size of queue, which is the number of nodes in the current level
            vector<int> curLevel; // Store the result per level
            for (int i = 0; i < n; ++i) {
                TreeNode* temp = q.front();
                q.pop();
                curLevel.push_back(temp -> val);
                if (temp -> left) q.push(temp -> left);
                if (temp -> right) q.push(temp -> right);
            }
            res.push_back(curLevel);
        }
        return res;
    }
};
```



#### [433. 最小基因变化](https://leetcode-cn.com/problems/minimum-genetic-mutation/)

使用队列 `queue` 进行 BFS：

```cpp
class Solution {
public:
    int cnt_diff (string s1, string s2) { // 记录两个字符串之间的距离
        int cnt = 0;
        for (int i = 0; i < s1.size(); ++i) {
            if (s1[i] != s2[i]) cnt++;
        }
        return cnt;
    }
    int minMutation(string start, string end, vector<string>& bank) {
        int res = 0;
        queue<string> q;
        unordered_map<string, int> visited; // whether the string has been visited
        q.push(start);
        while (!q.empty()) {
            int n = q.size();
            for (int i = 0; i < n; ++i) {
                string curString = q.front();
                visited[curString] = 1;
                q.pop();
                
                // 易错点
                if (curString == end) return res; // 队列中有目标串，则返回 res
                
                for (string b : bank) {
                    if (cnt_diff(curString, b) == 1 && !visited[b]) {
                        // if (b == end) return ++res;
                        q.push(b);
                        // res++;
                    }    
                }
            }
            // for loop 对当前队列中的所有 string 找距离为 1 且在 bank 中的 string，全部找完才 res++
            res++;
        }
        return -1; // 队列中没有目标串，则返回 -1
    }
};
```



#### [22. 括号生成](https://leetcode-cn.com/problems/generate-parentheses/)

递归，`n` 表示左括号和右括号的数目

> 其实也是深度优先搜索，因为递归的状态树中每一个节点都可以分叉处两个分支

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
        if (left < n) {
            generate(left + 1, right, n, s + "(");
        }

        if (left > right) {
            generate(left, right + 1, n, s + ")");
        }
        

        // reverse states

    }

};
```



#### [515. 在每个树行中找最大值](https://leetcode-cn.com/problems/find-largest-value-in-each-tree-row/)

> 与 [102. 二叉树的层序遍历](https://leetcode-cn.com/problems/binary-tree-level-order-traversal/) 类似，先 BFS 层序遍历，然后记录下每一层中的最大值

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
    vector<int> largestValues(TreeNode* root) {
        if (!root) return {};
        vector<int> res;
        queue<TreeNode*> q;
        q.push(root);
        
        while (!q.empty()) {
            int n = q.size();
            int max = INT_MIN;
            
            for (int i = 0; i < n; ++i) {
                TreeNode* tmp = q.front();
                q.pop();
                if (tmp -> val > max) {
                    max = tmp -> val;    
                }
                if (tmp -> left) q.push(tmp -> left);
                if (tmp -> right) q.push(tmp -> right);
            }
            res.push_back(max);
        }
        return res;
    }
};
```



#### [200. 岛屿数量](https://leetcode-cn.com/problems/number-of-islands/)

解题思路：

即深度优先搜索。循环遍历二维数组中的元素，每当碰到 `1` 说明有岛屿存在，岛屿数量 `+1`，并将 `1` 和其周围上下左右的 `1` 都变为 `0`，直到二维数组中全部为 `0`

```cpp
class Solution {
public:
    void DFSMarking(vector<vector<char>>& grid, int i, int j) {
        if (i < 0 || j < 0 || i >= grid.size() || j >= grid[0].size() || grid[i][j] == '0') 
            return ;
        grid[i][j] = '0';
        DFSMarking(grid, i + 1, j);
        DFSMarking(grid, i - 1, j);
        DFSMarking(grid, i, j + 1);
        DFSMarking(grid, i, j - 1);
    }

    int numIslands(vector<vector<char>>& grid) {
        int count = 0;
        for (int i = 0; i < grid.size(); ++i) {
            for (int j = 0; j < grid[i].size(); ++j) {
                if (grid[i][j] == '1') {
                    count++;
                    DFSMarking(grid, i, j);
                }
            }
        }
        return count;
    }
};
```



#### [127. 单词接龙](https://leetcode-cn.com/problems/word-ladder/)

1. **Typical BFS** structure.

```cpp
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> dict(wordList.begin(), wordList.end());
        queue<string> todo;
        todo.push(beginWord);
        int ladder = 1;
        while (!todo.empty()) {
            int n = todo.size();
            for (int i = 0; i < n; i++) {
                string word = todo.front();
                todo.pop();
                if (word == endWord) {
                    return ladder;
                }
                dict.erase(word);
                for (int j = 0; j < word.size(); j++) {
                    char c = word[j];
                    for (int k = 0; k < 26; k++) {
                        word[j] = 'a' + k;
                        if (dict.find(word) != dict.end()) {
                            todo.push(word);
                        }
                     }
                    word[j] = c;
                }
            }
            ladder++;
        }
        return 0;
    }
};
```

>  Cannot be accepted for time limit



**Optimization**:

```cpp
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> dict(wordList.begin(), wordList.end());
        queue<string> todo;
        todo.push(beginWord);
        int ladder = 1;
        while (!todo.empty()) {
            int n = todo.size();
            for (int i = 0; i < n; ++i) {
                string word = todo.front();
                todo.pop();
                if (word == endWord) {
                    return ladder;
                }
                // dict.erase(word);
                for (int j = 0; j < word.size(); j++) {
                    char c = word[j]; // 提前保存每个位置的字符
                    for (int k = 0; k < 26; ++k) {
                        word[j] = 'a' + k;
                        if (dict.find(word) != dict.end()) {
                            todo.push(word);
                            dict.erase(word); // optimize here for AC!!!
                        }
                    }
                    word[j] = c; // 还原当前位置的字符， 否则下次循环，word就变了
                }
            }
            ladder++;
        }
        return 0;
    }
};
```

> A small optimization to above can be erasing a word when pushing it into the queue. Because the length of wordList always is huge.



2. **Two-end BFS** solution:

双向BFS搜索示意图：其搜索的单词数量更少

![双向BFS搜索示意图](https://pic.leetcode-cn.com/38dc5897de2b554ea606a92c5eada14b0e0030195334e9fd65943ed6d0f77c1d-image.png)

* 已知目标顶点的情况下，可以分别从起点和终点执行广度优先遍历，直到遍历的部分有交集，这种方式遍历的单词数量会更小一些。
* 面试一般不需掌握双向广度优先搜索。
* 更合理的做法是：**每次从单词数量小的集合开始扩散**
* 等价于单向 BFS 里使用队列

使用条件：

* 明确的知道起点和终点是什么

核心思想：

* 需要两个 set：head、tail，分别存放从起点到终点、从终点到起点所匹配到的单词。
* 需要两个 set 指针，用于交换两个 set 的内容。
* 不要在意这些单词是否在一条路径上，set 只需控制路径的可能性。

```cpp
class Solution {
public:
    int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
        unordered_set<string> dict(wordList.begin(), wordList.end()), head, tail, *phead, *ptail;
        if (dict.find(endWord) == dict.end()) {
            // 处理 endWord 不在 wordList 中的特殊情况
            return 0;
        }
        head.insert(beginWord);
        tail.insert(endWord);
        int ladder = 2;
        while (!head.empty() && !tail.empty()) { // 当两个 set 都空时，才说明线索断了，所以要用 &&
            if (head.size() < tail.size()) { // 每次都选择较短的 set 进行搜索，可以保证考虑的情况尽量的少
                phead = &head;
                ptail = &tail;
            } else {
                phead = &tail;
                ptail = &head;
            }
            // 存储要进行扩散的下一层元素，在扩散完成后，会成为新的 phead 指向的 set
            unordered_set<string> temp; // 当temp最终为空时，就是线索断了
            for (auto it = phead -> begin(); it != phead -> end(); it++) {    
                string word = *it;
                for (int i = 0; i < word.size(); i++) {
                    char t = word[i];
                    for (int j = 0; j < 26; j++) {
                        word[i] = 'a' + j;
                        if (ptail -> find(word) != ptail -> end()) {
                            return ladder; // 当前phead中的元素已经出现在了ptail中，即找到了
                        }
                        if (dict.find(word) != dict.end()) {
                            temp.insert(word); // 将可能的选择 insert 到 temp
                            dict.erase(word); // dict 中 erase，避免重复选择
                        }
                    }
                    word[i] = t; // 字符复原，下次再用
                }
            }
            ladder++;
            // 这一行代表向外扩散了一层，用新的扩散的一层取代上一层，再进行扩散
            phead -> swap(temp); // 想比直接赋值，swap 是交换两个 set 的引用，更快
        }
        return 0;
    }
};
```



#### [529. 扫雷游戏](https://leetcode-cn.com/problems/minesweeper/)

DFS（深度优先、递归）：

```cpp
class Solution {
public:
    vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click) {
        int n = board.size(), m = board[0].size();
        int row = click[0], col = click[1];
        if (board[row][col] == 'M') { // 雷被挖出，结束，返回 board
            board[row][col] = 'X';
            return board;
        }

        vector<vector<int>> dirs = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}; // 坐标转换的方向
        int num = 0; // 当前坐标周围雷的数量
        for (auto dir : dirs) {
            int new_row = row + dir[0];
            int new_col = col + dir[1];
            if (new_row >= 0 && new_row < n && new_col >= 0 && new_col < m && board[new_row][new_col] == 'M') num++; 
        }

        if (num > 0) {
            board[row][col] = num + '0'; // board 中每个元素都是字符
            return board; // 如果不返回的话，下面的函数会继续执行，又修改为'B'了
        }
		// 也可以把下面的两句话放到 else{} 里面，这样 if (num > 0) 也会调用最后的 return board;
        board[row][col] = 'B'; // 周围没有雷
        for (auto dir : dirs) {
            int new_row = row + dir[0];
            int new_col = col + dir[1];
            if (new_row >= 0 && new_row < n && new_col >= 0 && new_col < m && board[new_row][new_col] == 'E') {
                vector<int> next_click = {new_row, new_col};
                updateBoard(board, next_click);
            } 
        }
        return board;
    }
};
```



### 贪心算法 Greedy

贪心算法是一种在每一步选择中都采取在当前状态下最好或最优的选择，从而希望导致结果是全局最好或最优的算法。

贪心可以解决一些最优化的问题，如：求图中的最小生成树、求哈夫曼编码等（面试中已很少出现）。然而对于工程和生活中的问题，贪心法一般不能得到我们想要的答案。

一旦一个问题可以通过贪心法来解决，那么贪心法一般是解决这个问题的最好方法。

由于贪心法的高效性以及其所求得的答案比较接近最优结果，贪心法也可以用作辅助算法或者直接解决一些要求结果不特别准确的问题。

**贪心算法和动态规划的不同**：

* 贪心算法它对每个子问题的解决方案都做出选择，不能回退。
* 动态规划则会保存以前的运算结果，并根据以前的结果对当前进行选择，有回退的功能。

> [动态规划定义](https://zh.wikipedia.org/wiki/%E5%8A%A8%E6%80%81%E8%A7%84%E5%88%92)

**贪心**：当下做局部最优判断

**回溯**：能够回退

**动态规划**：最优判断 + 回退



有些问题贪心不一定能够得到最优的结果，比如 [Coin Change 特别版本](https://leetcode-cn.com/problems/coin-change/):

当硬币可选集合固定：`Coins = [20, 10, 5, 1]`

求最少可以几个硬币拼出总数。

比如 `total = 36`

因为 `Coins` 里面的硬币都是倍数关系，所以可以用贪心算法。

![Cb0cK.png](https://wx2.sbimg.cn/2020/07/18/Cb0cK.md.png)

![Cb2YG.png](https://wx1.sbimg.cn/2020/07/18/Cb2YG.md.png)

![CbCVT.png](https://wx2.sbimg.cn/2020/07/18/CbCVT.md.png)



**何种情况下可以用贪心算法：**

即具有最优子结构

* 问题能够分解成子问题来解决。
* 子问题的最优解能够递推到最终问题的最优解。



### 贪心算法实战

#### [455. 分发饼干](https://leetcode-cn.com/problems/assign-cookies/)

贪心法：

```cpp
int findContentChildren(vector<int>& g, vector<int>& s) {
    sort(g.begin(), g.end());
    sort(s.begin(), s.end());
    int res = 0;
    int i = 0, j = 0;
    while (i != g.size() && j != s.size()) {
        if (g[i] <= s[j]) {
            res++;
            i++;
            j++;
        } else {
            j++;
        }
    }
    return res;
}
```

> * 时间复杂度：$O(m\log m)$，其中，$m=max(n_1, n_2)$，$n_1, n_2$分别为$g, s$数组的大小
>   `sort` 算法的复杂度为$O(n\log n)$,
>   `while` 循环的复杂度是$O(n_1+n_2)$,
>   所以总的时间复杂度是$O(m\log m)$。
>
> * 空间复杂度：$O(1)$，只开辟了常数变量的大小。



#### [122. 买卖股票的最佳时机 II](https://leetcode-cn.com/problems/best-time-to-buy-and-sell-stock-ii/)

贪心算法：

从第` i `天（这里 `i >= 1`）开始，与第 `i - 1` 的股价进行比较，如果股价有上升（严格上升），就将升高的股价（ `prices[i] - prices[i- 1]` ）记入总利润，按照这种算法，得到的结果就是符合题意的最大利润。

几点说明：

* 该算法仅可以用于计算，但计算的过程并不是真正交易的过程，但可以用贪心算法计算题目要求的最大利润。

下面说明这个等价性：以 [1, 2, 3, 4] 为例，这 4 天的股价依次上升，按照贪心算法，得到的最大利润是：

```cpp
res = (prices[3] - prices[2]) + (prices[2] - prices[1]) + (prices[1] - prices[0])
    = prices[3] - prices[0]
```

仔细观察上面的式子，按照贪心算法，在索引为 1、2、3 的这三天，我们做的操作应该是买进昨天的，卖出今天的，虽然这种操作题目并不允许，但是它等价于：“在索引为 0 的那一天买入，在索引为 3 的那一天卖出”。

这道题 “贪心” 的地方在于，对于 “今天的股价 - 昨天的股价”，得到的结果有 3 种可能：（1）正数（2）`0`（3）负数

贪心算法的决策是：**只加正数**。

```cpp
int maxProfit(vector<int>& prices) {
    int res = 0;
    int n = prices.size();
    for (int i = 1; i < n; ++i) {
        res += max(prices[i] - prices[i - 1], 0); // 只加正数
    }
    return res;
}
```



#### [55. 跳跃游戏](https://leetcode-cn.com/problems/jump-game/)

贪心算法：

从后往前贪心，`endReachable` 记录能够到达的最后位置的下标，往前移动下标的位置，看最后是否为 `0`。

```cpp
bool canJump(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return false;
    int endReachable = n - 1;
    for (int i = n - 1; i >= 0; --i) {
        if (nums[i] + i >= endReachable) { // nums[i] + i 中的 i 可以理解成已走过的距离，再加上 nums[i] 步，看能否到达最后的位置
            endReachable = i;
        }
    }
    return endReachable == 0;
}
```



贪心算法2：

* 如果某一个作为 **起跳点** 的格子可以跳跃的距离是 3，那么表示后面 3 个格子都可以作为 **起跳点**。
* 可以对每一个能作为 **起跳点** 的格子都尝试跳一次，把 能跳到 **最远的距离** 不断更新。
* 如果可以一直跳到最后，就成功了。

```cpp
bool canJump(vector<int>& nums) {
    int n = nums.size();
    int longestJump = 0;
    for (int i = 0; i < n; ++i) {
        if (i > longestJump) return false;
        longestJump = max(longestJump, nums[i] + i);
        if (longestJump >= n - 1) return true; // 已经可以跳到最后
    }
    return false;
}
```



#### [860. 柠檬水找零](https://leetcode-cn.com/problems/lemonade-change/)

运用了**贪心的思想**：

优先使用1张10元和1张5元的给20找零，而不是用3张5元的找零。

使用 `unordered_map` 存储5元和10元的数量，20的找不出去，所以不用存。

```cpp
bool lemonadeChange(vector<int>& bills) {
    int n = bills.size();
    if (n == 0) return true;
    unordered_map<int, int> map;
    for (int i = 0; i < n; ++i) {
        if (bills[i] == 5) {
            map[5]++;
        } else if (bills[i] == 10) {
            map[5]--;
            map[10]++;
        } else if (bills[i] == 20 && map[10] > 0) {
            map[5]--;
            map[10]--;
        } else { // bills[i] == 20 && map[10] == 0
            map[5] -= 3;
        }
        if (map[5] < 0) return false;
    }
    return true;
}
```

更快的方法（中间如果有找不开的情况，直接返回 `false`）：

```cpp
bool lemonadeChange(vector<int>& bills) {
    int n = bills.size();
    if (n == 0) return true;
    // 可以直接设置成如下形式
    // int c5 = 0, c10 = 0;
    unordered_map<int, int> map;
    for (int i = 0; i < n; ++i) {
        if (bills[i] == 5) {
            map[5]++;
        } else if (bills[i] == 10) {
            map[10]++;
            if (map[5] <= 0) return false; // 5 不够，10 找不开
            map[5]--;
        } else {
            if (map[10] >= 1 && map[5] >= 1) {
                map[10]--;
                map[5]--;
            } 
            else if (map[5] >= 3) map[5] -= 3; // 10 不够，用 5 找
            else return false; // 20 找不开
        }
    }
    return true;
}
```



### 二分查找

二分查找的前提：

* 目标函数的单调性（单调递增或者递减）
* 存在上下界（bounded）
* 能够通过索引访问（index accessible）

> 对于无序的数组，无法使用二分查找，只能从头遍历

**代码模板**：

```python
left, right = 0, len(array) - 1
while left <= right:
    mid = (left + right) / 2
    if array[mid] == target:
        # find the target!
        break or return result
    elif array[mid] < target:
        left = mid + 1
    else:
        right = mid - 1
```



### 二分查找实战

#### 二分查找C++代码模板

```cpp
int binarySearch(const vector<int>& nums, int target) {
	int left = 0, right = (int)nums.size()-1;
	
	while (left <= right) {
		int mid = left + (right - left)/ 2;
		if (nums[mid] == target) return mid;
		else if (nums[mid] < target) left = mid + 1;
		else right = mid - 1;
	}
	
	return -1;
}
```



#### [69. x 的平方根](https://leetcode-cn.com/problems/sqrtx/)

**方法1：二分查找**

因为 $y=x^2, (x>0)$: 

* 抛物线，在y轴右侧单调递增；
* 存在上下界

```cpp
int mySqrt(int x) {
    if (x == 0 || x == 1) return x;
    long left = 1, right = x;
    while (left <= right) {
        long mid = left + (right - left) / 2; // 防止越界
        if (mid * mid > x) right = mid - 1;
        else left = mid + 1;
    }
    return (int)right; // 具体是right还是left，写两个例子看一下
}
```

> `mid * mid` 会越界，所以要用 `long`

* 早期的操作系统是16位系统，
  `int` 用二字节表示，范围是 -32768~32767；
  `long` 用4字节表示，范围是 -2147483648~2147483647。
* 后来发展到32位操作系统，
  `int` 用4字节表示，与 `long` 相同。
* 目前的操作系统已发展到64位操作系统，但因程序编译工艺的不同，两者表现出不同的差别：
  32位编译系统：`int` 占四字节，与`long `相同。
  64位编译系统：`int ` 占四字节，`long` 占8字节，`long` 数据范围变为：$-2^{63} \sim 2^{63}-1$
* 在标准中，并没有规定 `long` 一定要比 `int` 长，也没有规定 `short` 要比 `int` 短。
  标准是这么说的: 长整型至少和整型一样长，整型至少和短整型一样长。
  这个的规则同样适用于浮点型 `long double` 至少和 `double` 一样长，`double` 至少和 `float` 一样长。

**`short` <= `int` <= `long`**

```cpp
short int  2个字节

int 2/4字节

long 4/8字节

long long 8字节
```



**方法2：牛顿迭代法**

写法1：

```cpp
int mySqrt(int x) {
    if (x == 0) return 0; // 要先判断0，否则x=0时while循环会超时
    double cur = x;
    while (true) {
        double pre = cur;
        cur = (cur + x / cur) / 2;
        if (abs(cur - pre) < 1e-6) return int(cur); 
    }
}
```

写法2：

```cpp
int mySqrt(int x) {
    long r = x;
    while (r * r > x) {
        r = (r + (x / r)) / 2;
    }
    return int(r);
}
```



#### [33. 搜索旋转排序数组](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/)

暴力：还原（可以用二分查找来找，$O(\log n)$），找到第一个无序元素的位置 -> 变为升序数组 -> 二分查找

正解：二分查找

判断时多加一个条件进行判断向左还是向右，即分段的二分查找

```cpp
int search(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1, mid;
    while (left <= right) {
        mid = left + (right - left) / 2;
        if (nums[mid] == target) return mid;
        // 先根据nums[mid]与nums[left]的关系判断mid是在左段还是右段
        if (nums[mid] >= nums[left]) { // mid 在左半段
            // 再判断target是在mid的左边还是右边，从而调整左右边界
            if (target >= nums[left] && target < nums[mid]) { // 在 mid 左边，两个条件都要成立
                right = mid - 1;
            } else {
                left = mid + 1;
            }
        } else { // mid 在右半段
            if (target <= nums[right] && target > nums[mid]) { // 在 mid 左边
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
    }
    return -1;
}
```



#### [153. 寻找旋转排序数组中的最小值](https://leetcode-cn.com/problems/find-minimum-in-rotated-sorted-array/)

> 此题可以用于寻找半有序数组中无序的地方

二分查找法：

```cpp
int findMin(vector<int>& nums) {
    int left = 0, right = nums.size() - 1, mid;
    while (left < right) {
        mid = left + (right - left) / 2; // mid 更靠近左边 left
        if (nums[mid] > nums[right]) { // mid 在左半段
            left = mid + 1;
        } else { // mid 在右半段
            right = mid; // 因为此段代码中没有判断nums[mid]等于目标值的逻辑，所以如果写成right = mid - 1;会将 mid遗漏掉
        }
    }
    return nums[left];
}
```

解释：

单调递增的序列：

```
            *
          *
        *
      *
    *
```

做了旋转：

```
  *
*
        *
      *
    *
```


用二分法查找，需要始终将目标值（这里是最小值）套住，并不断收缩左边界或右边界。

左、中、右三个位置的值相比较，有以下几种情况：

1. 左值 < 中值, 中值 < 右值 ：没有旋转，最小值在**最左边**，可以**收缩右边界**

```
        右
    中
左
```

2. 左值 > 中值, 中值 < 右值 ：有旋转，最小值在**左半边**，可以**收缩右边界**

 ```
左       
        右
    中
 ```

3. 左值 < 中值, 中值 > 右值 ：有旋转，最小值在**右半边**，可以**收缩左边界**

```
    中
左 
        右
```

4. 左值 > 中值, 中值 > 右值 ：单调递减，不可能出现

 ```
左
    中
        右
 ```


分析前面三种可能的情况，会发现情况1、2是一类，情况3是另一类。

如果中值 < 右值，则最小值在左半边，可以收缩右边界。
如果中值 > 右值，则最小值在右半边，可以收缩左边界。

通过比较中值与右值，可以确定最小值的位置范围，从而决定边界收缩的方向。

而情况1与情况3都是左值 < 中值，但是最小值位置范围却不同，这说明，如果只比较左值与中值，不能确定最小值的位置范围。

所以我们需要通过比较**中值**与**右值**来确定最小值的位置范围，进而确定边界收缩的方向。



为什么比较`mid`与`right`而不比较`mid`与`left`？

具体原因前面已经分析过了，简单讲就是因为我们**找最小值**，要**偏向左找**，目标值右边的情况会比较简单，容易区分，所以比较`mid`与`right`而不比较`mid`与`left`。

那么能不能通过比较`mid`与`left`来解决问题？

能，转换思路，**不直接找最小值，而是先找最大值，最大值偏右**，可以通过比较`mid`与`left`来找到最大值，**最大值向右移动一位就是最小值了**（需要考虑最大值在最右边的情况，右移一位后对数组长度取余）。



以下是先找最大值的代码，最大值右移一位取余就是最小值：

```cpp
class Solution {
public:
    int findMin(vector<int>& nums) {
        int left = 0;
        int right = nums.size() - 1;
        while (left < right) {
            int mid = left + (right - left + 1) / 2;   /* 先加一再除，mid更靠近右边的right */
            if (nums[left] < nums[mid]) {
                left = mid;                            /* 向右移动左边界 */
            } else if (nums[left] > nums[mid]) {
                right = mid - 1;                       /* 向左移动右边界 */
            }
        }
        return nums[(right + 1) % nums.size()];    /* 最大值向右移动一位就是最小值了（需要考虑最大值在最右边的情况，右移一位后对数组长度取余） */
    }
};
```































