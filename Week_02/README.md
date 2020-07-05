# 第二周总结

[TOC]

> 养成收藏优秀代码的习惯
>
> [Markdown 中使用各种特殊符号](https://blog.csdn.net/qq_40942329/article/details/78724322)
>
> [数据结构在线演示网站1](https://visualgo.net/zh)
>
> [数据结构在线演示网站2](https://www.cs.usfca.edu/~galles/visualization/Algorithms.html)

### Ⅰ哈希表（Hash table）、映射（map）、集合（set）

#### 哈希表

* 哈希表（Hash table），也叫散列表，是根据关键码值（Key value）而直接进行访问的数据结构。

* 它通过把关键码值映射到表中一个位置（即下标index）来访问记录，以加快查找的速度。

* 这个映射函数叫做散列函数（Hash Function），存放记录的数组叫作哈希表（或散列表）。

> 工程实践：电话号码簿、用户信息表、缓存（LRU Cache）、键值对存储（Redis）

**哈希碰撞（Hash Collisions）**

对于不同的要存储的数据，经过哈希函数之后会得到一个相同的值。此时，再增加一个维度，在当前位置拉出一个链表，即**拉链式解决冲突法**

> 如果链表很长的话，会将Hash的查询效率从 $O(1)$ 退化成 $O(n)$，所以要设计合理，使哈希函数的碰撞概率很小。

平均时刻，哈希函数的查询仍是 $O(1)$ 的。

现实中大部分位置是完美哈希的，只有一个元素，只有少数位置会发生冲突。Hash 的 size 我们可以设计的很大。

**时间复杂度**

查询、添加、删除 - $O(1)$

**工程中的应用**

* **Map**：key-value 对，key 不重复
* **Set**：不重复元素的集合

如果 Hash table 是 TreeMap 或 TreeSet 的话，都是使用红黑树来实现的，所以所有操作的时间复杂度都为 $O(\log n)$



### Ⅱ 哈希表实战

C++ 中的 `std::map`是用 red-black trees 实现的，内部通过`Compare`函数对键进行了排序，所以**搜索、删除、插入**操作都是 $O(\log n)$ 的时间复杂度。

`std::unordered_map`是用 hash table 来实现的，所以**搜索、删除、插入**操作都是 $O(1)$ 的时间复杂度。



#### `std::unordered_map`的**内置函数**

`iterator find(const Key& key )`：

* 找到与当前的 key 相等的 key
* Return value：返回与当前 key 相等的 key 的元素的迭代器。如果没有这样的元素，则返回 past-the-end 迭代器

代码示例：

```cpp
#include <iostream>
#include <unordered_map>
 
int main()
{  
// simple comparison demo
    std::unordered_map<int,char> example = {{1,'a'},{2,'b'}};
 
    auto search = example.find(2);
    if (search != example.end()) {
        std::cout << "Found " << search -> first << " " << search -> second << '\n';
    } else {
        std::cout << "Not found\n";
    }
}
```

Output:

```cpp
Found 2 b
```

`iterator end()`：

* 返回对 unordered_map 最后一个元素后面的元素的迭代器
* 这个元素相当于占位符，试图访问它会导致未定义的行为
* Return value：Iterator to the element following the last element.



#### 1. 两数之和

使用 hash table 实现的代码：

```cpp
vector<int> twoSum(vector<int>& nums, int target) {
    //Key is the number and value is its index in the vector.
    unordered_map<int, int> hash;
    vector<int> result;
    for (int i = 0; i < nums.size(); ++i) {
        int numberToFind = target - nums[i];

        //if numberToFind is found in map, return them
        if (hash.find(numberToFind) != hash.end()) {
            result.push_back(hash[numberToFind]);
            result.push_back(i);
            return result;
        }

        //number was not found. Put it in the map.
        hash[nums[i]] = i;
    }
    return result;
}
```

- 时间复杂度：$O(n)$，只遍历了含有 n 个元素的列表一次，在表中进行每次查找只花费 $O(1)$ 的时间。

- 空间复杂度：$O(n)$，使用哈希表存储，最多存储 n 个元素。

> 题目要求每个位置的元素不可以重复使用，而上面的代码恰好不会出现重复使用的情况。



#### 242. 有效的字母异位词

* Clarification (审题、沟通题目)：如 大小写是否敏感、什么是异位词
* Possible solution -> optimal (time & space)
* Code (写代码)
* Test cases (测试样例)

1. 暴力：
   先 sort（排序函数可以直接调用），看 sorted_str 是否相等 $O(n\log n)$。Cpp 中的排序函数为`void sort( RandomIt first, RandomIt last )`

   > Sorts the elements in the range [first, last) in non-descending order. The order of equal elements is not guaranteed to be preserved.

   sort 函数示例，defined in header \<algorithm\> 

   ```cpp
   #include <algorithm>
   #include <array>
   #include <iostream>
   int main()
   {
       std::array<int, 10> s = {5, 7, 4, 2, 8, 6, 1, 9, 0, 3}; 
    
       // sort using the default operator<
       std::sort(s.begin(), s.end());
       for (auto a : s) {
           std::cout << a << " ";
       }   
       std::cout << '\n';
   }
   
```
   
**提交代码**：
   
   ```cpp
   bool isAnagram(string s, string t) {
       if (s.size() != t.size()) {
           return false;
       }
       sort(s.begin(), s.end());
       sort(t.begin(), t.end());
       return s == t;
   }
```
   
   * 时间复杂度：$O(n\log n)$，排序时间占主导，Cpp 中排序函数 sort 的时间复杂度为 $O(n\log n)$
* 空间复杂度：$O(1)$
   
2. hash, map:

   * 统计每个字符的频次，看频次是否相等
   * 第一个单词，碰到相应的字母 +1，第二个单词碰到相应的字母 -1，最后看 map 是否为空，或者所有的计数器是否都为 0
   * 定义一个长度为 256 (因为 ASCII 码是 0 到 255)的数组进行计数。（简化了的哈希表）

   使用内置的 `unordered_map`：

   > For each letter in `s`, it increases the counter by `1` while for each letter in `t`, it decreases the counter by `1`

   **提交代码**：

   ```cpp
   bool isAnagram(string s, string t) {
       if (s.size() != t.size()) {
           return false;
       }
       int n = s.size();
       unordered_map<char, int> counts;
       for (int i = 0; i < n; ++i) {
           counts[s[i]]++;
           counts[t[i]]--;
       }
   
       for (auto count : counts) {
           if (count.second) { // 检索 map 中的 value 值是否有非 0 元素
               return false;
           }
       }
   
       return true;
   }
   ```

   * 时间复杂度：$O(n)$
   * 空间复杂度：$O(n)$，哈希存储

   

   题目只包含小写字母，可以使用**数组**来模拟 map，使代码加速。

   **提交代码**：
   
   ```cpp
   bool isAnagram(string s, string t) {
       if (s.size() != t.size()) {
           return false;
       }
       int n = s.size();
       int counts[26] = {0};
       for (int i = 0; i < n; ++i) {
           counts[s[i] - 'a']++;
           counts[t[i] - 'a']--;
       }
   
       for (int i = 0; i < 26; ++i) {
           if (counts[i]) return false;
       }
   
       return true;
}
   ```
   
   * 时间复杂度：$O(n)$
   * 空间复杂度：$O(1)$

**进阶**：

如果输入字符串包含 unicode 字符怎么办？你能否调整你的解法来应对这种情况？

**解答**：

使用哈希表而不是固定大小的计数器。想象一下，分配一个大的数组来适应整个 Unicode 字符范围，这个范围可能超过 100万。哈希表是一种更通用的解决方案，可以适应任何字符范围。



#### 49. 字母异位词分组

使用 `unordered_map`

> Use an `unordered_map` to group the strings by their sorted counterparts. Use the sorted string as the key and all anagram strings as the value.
>
> `unordered_map`第一维是排序后的字符串，第二维是原始的字符串

**提交代码**：

```cpp
vector<vector<string>> groupAnagrams(vector<string>& strs) {
    unordered_map<string, vector<string>> mp;
    for (string s : strs) {
        string t = s;
        sort(t.begin(), t.end());
        mp[t].push_back(s); // 第二维是向量，所以 .push_back()
    }
    vector<vector<string>> anagrams; // 第二维是字符串
    for (auto p : mp) {
        anagrams.push_back(p.second);
    }
    return anagrams;
}
```

* 时间复杂度：$O(nk\log k)$，其中 n 是`strs`的长度，而 k 是`strs`中的最长字符串的长度。外部循环的复杂度为 $O(n)$，内部排序的复杂度为 $O(k\log k)$
* 空间复杂度：$O(nk)$



题目只包含小写的字母，因此我们可以用一个数组进行计数排序，提高一点速度

**提交代码**：

```cpp
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;
        for (string s : strs) {
            mp[strSort(s)].push_back(s); // 利用计数数组对 s 排序，返回排好序的结果 strSort(s)
        }

        vector<vector<string>> anagrams;
        for (auto p : mp) {
            anagrams.push_back(p.second);
        }
        return anagrams;
    }
private:
    string strSort(string s) {
        int counter[26] = {0};
        for (char c : s) {
            counter[c - 'a']++;
        }
        string t;
        for (int c = 0; c < 26; ++c) {
            // string(int n,char c);  用 n 个字符 c 初始化
            t += string(counter[c], c + 'a'); 
        }
        return t;
    }
};
```

> 记住 Cpp 中 string 是个类
>
> `string(int n,char c);`  构造函数，指用 n 个字符 c 初始化字符串



### III 树（Tree）、二叉树（Binary Tree）、二叉搜索树（Binary Search Tree）

> 一维数据结构：加速 --> 升维

#### 树（Tree）

Linked List 是特殊化的 Tree

Tree 是特殊化的 Graph （如果树有环 --> 图）

![2j7UG.png](https://wx2.sbimg.cn/2020/06/29/2j7UG.png)



#### 二叉树（Binary Tree）

**树结点定义**：

```cpp
// Definition for a binary tree node.
struct TreeNode{
    int val;
    TreeNode *left;
    TreeNode *right;
    TreeNode(int x): val(x), left(NULL), right(NULL) {}
};
```

**二叉树的遍历**：基于递归

* 前序（Pre-order）：根-左-右
* 中序（In-order）：左-根-右
* 后序（Post-order）：左-右-根



#### 二叉搜索树（Binary Search Tree）

又称二叉排序树、有序二叉树（Ordered Binary Tree）、排序二叉树（Sorted Binary Tree），是指一棵空树或者具有下列性质的二叉树：

	1. 左子树上所有结点的值均小于它的根结点的值
 	2. 右子树上所有结点的值均大于它的根结点的值
 	3. 以此类推：左、右子树也分别为二叉搜索树

**中序遍历**：升序排列

**常见操作**：

* 查询 - $O(\log n)$
* 插入新结点（创建）- $O(\log n)$
* 删除 - $O(\log n)$

> 删除操作：叶子结点直接删除；非叶子结点一般是找大于当前结点的第一个结点，进行替换

[BST在线演示Demo](https://visualgo.net/zh/bst)

**思考题：**

树的面试题解法一般都是递归，为什么？

* 树没有便于循环的结构

> 用递归来解决的问题，一般具有如下特点：  
> 1. 可以被分解成重复的子问题；  
> 2. 子问题可以使用相同的算法来解决；  
> 3. 有明确的终止条件   
>
> 树符合这个条件，可以用递归实现



### Ⅳ 实战题目解析

> 递归本身并不存在效率差的问题，只要程序本身的算法复杂度没有写残即可。
>
> 比如 Fibonacci 数列如果只是傻递归，没有把中间结果储存起来的话，就会导致本身线性时间复杂度可以解决的问题需要指数级的时间复杂度才能解决。所以锅并不在递归。
>
> 我们认为递归和非递归的方式，递归可能会慢一点，因为系统要多开一些栈，如果递归的深度很深的话，递归是会比较慢。一般情况下，基于现在的计算机的存储方式和编译器对于递归特别是尾递归的优化，我们就直接认为递归和循环的效率是一样的。

#### 94. 二叉树的中序遍历

> 递归，即系统帮你创建一个栈，把要调用的函数和它相应的参数依次压入栈中

采用递归进行中序遍历，定义辅助函数实现递归

**提交代码**：

```cpp
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
```

* 时间复杂度：$O(n)$。递归函数 $T(n) = 2 \cdot T(\frac{n}{2})+O(1)$。
* 空间复杂度：最坏情况下需要空间$O(n)$，平均情况为$O(\log n)$。



基于栈的遍历，自己手动维护一个栈

```cpp
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
```



#### 144. 二叉树的前序遍历

> 前序遍历需要根结点先入栈

递归实现

**提交代码**：

```cpp
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
```



使用栈来模拟递归实现

**提交代码**：

```cpp
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
```



#### 589. N叉树的前序遍历

> 前序遍历需要根结点先入栈

递归实现

**提交代码**：

```cpp
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
```



使用栈来模拟递归实现

**提交代码**：

```cpp
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
```

> 该题与 144 题的**栈实现**的区别：while 循环中 if 语句部分对孩子结点的入栈处理上。
>
> 不过都是最右边的孩子结点先入栈，这样就能保证最右边的结点最后出栈。



#### 590. N叉树的后序遍历

递归

**提交代码**：

```cpp
class Solution {
public:
    vector<int> postorder(Node* root) {
        vector<int> res;
        traversal(root, res);
        return res;
    }
private:
    void traversal(Node* root, vector<int>& res) {
        if (!root) {
            return ;
        }
        for (Node* child : root -> children) {
            traversal(child, res);
        }
        res.push_back(root -> val);
    }
};
```



使用栈模拟递归实现（最后需要反转 vector）

**提交代码**：

```cpp
vector<int> postorder(Node* root) {
    vector<int> res;
    if (!root) {
        return res;
    }

    stack<Node*> stk;
    stk.push(root); 
    while (!stk.empty()) {
        Node* temp = stk.top(); // 先把栈顶元素取出来
        stk.pop();
        for (int i = 0; i < temp -> children.size(); i++) {
            stk.push(temp -> children[i]);
        }
        res.push_back(temp -> val); // 再决定栈顶元素 val 存储的位置
    }
    reverse(res.begin(), res.end()); // 核心，反转
    return res;
}
```



使用栈模拟递归实现（无需反转）

**提交代码**：

```cpp
vector<int> postorder(Node* root) {
    if (!root) {
        return vector<int>(); // 返回空 vector
    }
    stack<Node*> stk;
    Node *p = root;
    stk.push(p);
    list<int> res;
    
    while(!stk.empty()) {
        p = stk.top();
        stk.pop();
        res.push_front(p -> val); // 对链表进行头插
        for(int i = 0; i < p -> children.size(); ++i)
            stk.push(p -> children[i]);
    }
    return vector<int>(res.begin(), res.end());
}
```



#### 429. N叉树的层序遍历

使用队列 queue

**提交代码**：

```cpp
vector<vector<int>> levelOrder(Node* root) 
{
    if (!root) {
        return vector<vector<int>>(); // We could also "return {};" here thanks to C++11.
    } 
         
    vector<vector<int>> res;
    queue<Node*> q; 
    q.push(root);
    while (!q.empty()) {
        int size = q.size(); // Store the size of queue, which is the number of nodes in the current level
        vector<int> curLevel; // Store the result per level. 
        for (int i = 0; i < size; i++) { // For each node of the current level
            Node* tmp = q.front(); // Get the first node from the queue
            q.pop(); // Pop this node since we no longer need it.
            curLevel.push_back(tmp -> val); // Store values of tmp nodes
            for (auto n : tmp -> children) // Push every child node of the tmp node back to the queue. FIFO(first in first out)
                q.push(n); 
        }
        res.push_back(curLevel); // Store the current level values to res.
    }
    return res; 
}
```



### Ⅴ 堆和二叉堆

####  堆（Heap）和二叉堆（Binary Heap）的实现和特性

> 堆是一个抽象的数据结构，表示可以迅速地拿到一堆数里面的最大值或最小值，它并不是二叉堆。
>
> 二叉堆是堆（优先队列 priority_queue）的一种常见且简单的实现，但是并不是最优的实现，详细参照Wiki

* 可以迅速找到一堆数中的 **最大** 或者 **最小值**的数据结构（如找优先级最高等）

* 大顶堆/大根堆：根节点最大的堆
* 小顶堆/小根堆：根节点最小的堆
* 常见的堆：二叉堆、斐波那契堆（工业上应用的比较多，基于树实现的，时间复杂度和空间复杂度更好）



假设是一个大顶堆，常见操作（API）：

* find-max - $O(1)$
* delete-max - $O(\log n)$
* insert (create) - $O(\log n)$ or $O(1)$（斐波那契堆）

[Wiki：不同实现的比较](https://en.wikipedia.org/wiki/Heap_(data_structure))

![2Hc0A.png](https://wx2.sbimg.cn/2020/07/01/2Hc0A.png)

> 上图中绿色代表最好。
>
> 可以看到最好的实现方式是严格斐波那契堆。



##### 二叉堆的性质

* 通过完全二叉树来实现，实现相对容易

  > （注意：不是二叉搜索树），二叉搜索树也可以实现，只不过找最小值或最大值时的时间复杂度变为 $O(\log n)$，就不是 $O(1)$了。

* 二叉堆（大顶）满足下列的性质：

  * [性质一] 是一颗完全二叉树（除了最下面的叶子节点不是满的，其他层的节点都是满的）
  * [性质二] 树中任意节点的值总是 >= 其子节点的值（保证根最大）

[![2HdsV.md.png](https://wx1.sbimg.cn/2020/07/01/2HdsV.md.png)](https://sbimg.cn/image/2HdsV)



##### 二叉堆的实现细节

> [二叉堆的 Java 实现](https://shimo.im/docs/ShKvLVRnWwUpCGpZ/)

1. 二叉堆一般都是通过“数组”实现的。（因为二叉堆是完全二叉树）
   * 根节点（顶堆元素）是：a[0]
2. 假设“第一个元素”在数组中的索引是0的话，则父节点和子节点的位置关系如下：
   * 索引为 $i$ 的左孩子的索引是 $(2 × i + 1)$
   * 索引为 $i$ 的右孩子的索引是 $(2 × i + 2)$
   * 索引为 $i$ 的父节点的索引是 $\lfloor \frac{i - 1}{2} \rfloor$

[![2HJBO.md.png](https://wx2.sbimg.cn/2020/07/01/2HJBO.md.png)](https://sbimg.cn/image/2HJBO)



###### 插入操作 Insert

1. 新元素一律先插入到堆的尾部

2. 依次向上调整整个堆的结构（一直到根即可）

   > 函数名：HeapifyUp（向上调整），java代码实现如下

   ```java
   /**
    * Maintains the heap property while inserting an element.
    */
   private void heapifyUp(int i) {
       int insertValue = heap[i];
       while (i > 0 && insertValue > heap[parent(i)]) {
           heap[i] = heap[parent(i)];
           i = parent(i);
       }
       heap[i] = insertValue;
   }
   
   private int parent(int i) { // 返回 i 的父节点
       return (i - 1) / d;
   }
   ```

3. 时间复杂度即树的深度 - $O(\log n)$

Insert 的 java 实现：

```java
/**
 * Inserts new element in to heap
 * Complexity: O(log N)
 * As worst case scenario, we need to traverse till the root
 */
public void insert(int x) {
    if (isFull()) {
        throw new NoSuchElementException("Heap is full, No space to insert new element");
    }
    heap[heapSize] = x;
    heapSize ++;
    heapifyUp(heapSize - 1); // 调整堆尾元素
}
```

[![2lrYU.md.png](https://wx2.sbimg.cn/2020/07/04/2lrYU.md.png)](https://sbimg.cn/image/2lrYU)

[![2lOFd.md.png](https://wx1.sbimg.cn/2020/07/04/2lOFd.md.png)](https://sbimg.cn/image/2lOFd)

[![2lTG4.md.png](https://wx2.sbimg.cn/2020/07/04/2lTG4.md.png)](https://sbimg.cn/image/2lTG4)

[![2lywY.md.png](https://wx1.sbimg.cn/2020/07/04/2lywY.md.png)](https://sbimg.cn/image/2lywY)



###### 删除堆顶操作 Delete Max

1. 将堆尾元素替换到顶部（即堆尾元素被替代删除掉）

2. 依次从根部向下调整整个堆的结构（一直到堆尾即可）

   > 函数名：HeapifyDown（向下调整）

   ```java
   /**
    * Maintains the heap property while deleting an element.
    */
   private void heapifyDown(int i) {
       int child;
       int temp = heap[i];
       while (kthChild(i, 1) < heapSize) { // 看第一个孩子节点是否存在，或者至少有一个孩子节点存在；或者第一个孩子不存在，那么其他孩子节点也不存在
           child = maxChild(i); // 最大的儿子的下标
           if (temp >= heap[child]) {
               break;
           }
           heap[i] = heap[child];
           i = child;
       }
       heap[i] = temp;
   }
   
   private int kthChild(int i, int k) { // i 的第 k 个孩子
       return d * i + k;
   }
   ```

3. 时间复杂度即树的深度 - $O(\log n)$

Delete 的 java 实现：

```java
 /**
 * Deletes element at index x
 * Complexity: O(log N)
 */
public int delete(int x) {
    if (isEmpty()) {
        throw new NoSuchElementException("Heap is empty, No element to delete");
    }
    int maxElement = heap[x];
    heap[x] = heap[heapSize - 1]; // 堆尾元素给堆顶
    heapSize--;
    heapifyDown(x); // 调整堆顶元素
    return maxElement;
}
```

![2r4x6.png](https://wx1.sbimg.cn/2020/07/04/2r4x6.png)

![2rMXO.png](https://wx2.sbimg.cn/2020/07/04/2rMXO.png)

![2rXde.png](https://wx1.sbimg.cn/2020/07/04/2rXde.png)

![2rmZD.png](https://wx2.sbimg.cn/2020/07/04/2rmZD.png)

### Ⅵ 实战题目解析

#### 剑指 Offer 40. 最小的k个数

* 方法一：sort - $O(n\log n)$

  > 直接调用系统函数即可

* 方法二：heap - $O(n\log k)$

  > 比方法一 快一点，由于大根堆实时维护前 $k$ 小值 (堆的大小为 $k$)，所以插入删除都是 $O(\log k)$ 的时间复杂度，最坏情况下数组里 $n$ 个数都会插入，所以一共需要 $O(n\log k)$ 的时间复杂度

* 方法三：quick-sort



调用系统的 $std::sort$ 函数进行排序，再取前 k 个数。

**提交代码**：

```cpp
vector<int> getLeastNumbers(vector<int>& arr, int k) {
    vector<int> res;
    if (k == 0 || arr.size() == 0) {
        return {};
    }
    std::sort(arr.begin(), arr.end()); // O(nlog n)
    for (int i = 0; i < k; ++i) {
        res.push_back(arr[i]);
    }
    return res;
}
```

----



调用系统堆 $priority\_queue$ (事实上执行效率比方法一还低，不如直接手写个大顶堆)

> 我们用一个**大根堆**实时维护数组的前 $k$ 小值。首先将前 $k$ 个数插入大根堆中，随后从第 $k+1$ 个数开始遍历，如果当前遍历到的数比大根堆的堆顶的数要小，就把堆顶的数弹出，再插入当前遍历到的数。最后将大根堆里的数存入数组返回即可。

priority_queue 代码示例：

```cpp
#include <functional>
#include <queue>
#include <vector>
#include <iostream>
 
template<typename T> void print_queue(T& q) {
    while(!q.empty()) {
        std::cout << q.top() << " ";
        q.pop();
    }
    std::cout << '\n';
}
 
int main() {
    std::priority_queue<int> q;
 
    for(int n : {1,8,5,6,3,4,0,9,7,2})
        q.push(n);
 
    print_queue(q);
 
    // using std::greater<T> would cause the smallest element to appear as the top()
    std::priority_queue<int, std::vector<int>, std::greater<int> > q2;
 
    for(int n : {1,8,5,6,3,4,0,9,7,2})
        q2.push(n);
 
    print_queue(q2);
}

/** 
* Output:
* 9 8 7 6 5 4 3 2 1 0 
* 0 1 2 3 4 5 6 7 8 9 
*/
```



**提交代码**：

```cpp
vector<int> getLeastNumbers(vector<int>& arr, int k) {
    if (k == 0 || arr.size() == 0) {
        return {};
    }
    vector<int> res;
    priority_queue<int> Q;
    for (int i = 0; i < k; ++i) {
        Q.push(arr[i]);
    }

    for (int i = k; i < arr.size(); ++i) {
        if (Q.top() > arr[i]) {
            Q.pop();
            Q.push(arr[i]);
        }
    }

    while (!Q.empty()) {
        // res.push_back(Q.pop()); is wrong!
        res.push_back(Q.top());
        Q.pop(); // .pop() return null
    }
    return res;
}
```

------



手写堆，对于这道题，只涉及到**建堆**和**push**操作

> * 首先对 arr 数组的前 $k$ 个数建堆;
> * 然后从 $k$ 开始对剩下的数组进行遍历，每个元素都和堆顶元素进行比较；
> * 如果当前元素比堆顶元素大，则不处理（我们是最大堆）；如果比当前元素小则将堆顶元素置换为当前元素，并调整堆

**提交代码**：

```cpp
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        if (k == 0) return vector<int>(); // return {};
        vector<int> maxHeapVector(arr.begin(), arr.begin() + k); // 对数组中的 k 个元素建立大顶堆
        buildMaxHeap(maxHeapVector); // 调整为大顶堆
        for (int i = k; i < arr.size(); ++i) {
            // 对于比堆顶元素小的元素，置换掉对顶元素，并调整堆
            if (arr[i] < maxHeapVector[0]) {
                emplacePeek(maxHeapVector, arr[i]);
            }
        }
        return maxHeapVector;
    }

private:
    void buildMaxHeap(vector<int>& heap) {
        // 对于所有非叶子节点，从后往前依次下沉
        for (int i = parent(heap.size() - 1); i >= 0; --i) { // i 的父节点为 (i - 1) / 2
            heapifyDown(heap, i);
        }
    }

    void heapifyDown(vector<int>& heap, int index) {
        int parent = heap[index];
        int childIndex = kthChild(index, 1); // 左孩子的索引
        while (childIndex < heap.size()) {
            // 判断是否存在右孩子，并选出最大孩子的下标
            if (childIndex + 1 < heap.size() && heap[childIndex + 1] > heap[childIndex]){
                childIndex++;
            }
            // 判断父节点和子节点的大小关系
            if (parent >= heap[childIndex]) break;
            // 较大的节点上浮
            heap[index] = heap[childIndex];
            index = childIndex;
            childIndex = kthChild(index, 1);
        }
        heap[index] = parent;
    }

    int parent(int index) {
        return (index - 1) / 2;
    }

    int kthChild(int index, int k) {
        return 2 * index + k;
    }

    void emplacePeek(vector<int>& heap, int val) {
        heap[0] = val; // 置换掉堆顶元素
        heapifyDown(heap, 0);
    }
};
```



#### 239. 滑动窗口最大值

使用 priority_queue 来解决

**提交代码**：

```cpp
vector<int> maxSlidingWindow(vector<int>& nums, int k) { // 
    int n = nums.size();
    if (n == 0 || k == 0) {
        return {};
    }
	
    vector<int> res;
    priority_queue<pair<int, int>> heap; // 这的 pair<T1, T2> 即元组
    for (int i = 0; i < n; ++i) {
        // 将前面的窗口的最大值弹出去，以免后面窗口的最大值小于前面窗口的最大值
        while (!heap.empty() && heap.top().second <= i - k) { 
            heap.pop();
        }
        heap.push(make_pair(nums[i], i)); // res.push({nums[i], i});
        if (i >= k - 1) { // 从第一个窗口开始存储
            res.push_back(heap.top().first);
        }
    }
    return res;
}
```



#### [347. 前 K 个高频元素](https://leetcode-cn.com/problems/top-k-frequent-elements/)

调用系统的 priority_queue 及 unordered_map

**提交代码**：

```cpp
vector<int> topKFrequent(vector<int>& nums, int k) {
    // 通过 std::greater<T> 建立小顶堆
    priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> pq;
    unordered_map<int, int> cnt; // 计数

    for (auto num : nums) cnt[num]++;
    for (auto kv : cnt) {
        pq.push({kv.second, kv.first});
        while (pq.size() > k) pq.pop();
    }

    vector<int> res;
    while (!pq.empty()) {
        res.push_back(pq.top().second);
        pq.pop();
    }
    return res;
}
```

> 优先队列的定义：priority_queue<Type, Container, Functional>
>
> Type 就是数据类型，Container 就是容器类型（Container必须是用数组实现的容器，比如vector, deque 等等，但不能用 list。STL 里面默认用的是 vector），Functional 就是比较的方式，当需要用自定义的数据类型时才需要传入这三个参数，使用基本数据类型时，只需要传入数据类型，默认是大顶堆。

```cpp
// 对于基础类型 默认是大顶堆
priority_queue<int> a;
// 小顶堆
priority_queue<int, vector<int>, greater<int>> b;
```



#### [剑指 Offer 49. 丑数](https://leetcode-cn.com/problems/chou-shu-lcof/)

「小顶堆的方法是先存再排，dp的方法则是先排再存」

**动态规划**（三指针）

一个数字可以拆分成若干因子之积，那么我们也可以使用不同因子去构造这个数。

首先第一个数是 1，然后是

2\*1, 2\*2, 3\*1, 5\*1 ......

我们可以看出来每次都是前面某个数字乘以 2 或者 3 或者 5，取其中的最小者。

我们定义**三个指针 t2、t3、t5 分别指向要乘以 2、3、5 的数字**。dp[i] 表示第 i 个丑数。

那么 dp[t2] * 2、dp[t3] * 3 和 dp[t5] * 5 中的最小值就是下一个丑数。



**提交代码**：

```cpp
int nthUglyNumber(int n) {
    if (n <= 0) return false;
    if (n == 1) return true; // base case
    int t2 = 0, t3 = 0, t5 = 0;
    vector<int> dp(n);
    dp[0] = 1;
    for (int i = 1; i < n; ++i) {
        dp[i] = min(dp[t2] * 2, min(dp[t3] * 3, dp[t5] * 5));
        if (dp[i] == dp[t2] * 2) t2++;
        if (dp[i] == dp[t3] * 3) t3++;
        if (dp[i] == dp[t5] * 5) t5++;
    }
    return dp[n - 1];
}
```

---



小顶堆

**提交代码**：

```cpp
int nthUglyNumber(int n) {
    priority_queue<double, vector<double>, greater<double>> minHeap;
    // 若声明为 int 类型，则测试时会溢出，改为 double 则没事
    double res = 1;
    for (int i = 1; i < n; ++i) {
        minHeap.push(res * 2);
        minHeap.push(res * 3);
        minHeap.push(res * 5);
        res = minHeap.top();
        minHeap.pop();
        // 判断当前最小的元素是否有重复的
        while (!minHeap.empty() && res == minHeap.top()) {
            minHeap.pop();
        }
    }
    return res;
}
```



### Ⅶ 图

####  图的属性和特性

> 面试中很少考图相关的东西

##### 1. 图的属性

* Graph(V, E)
* V - vertex: 点
  * 度 - 入度和出度
  * 点与点之间：连通与否
* E - edge: 边
  * 有向和无向（单行线）
  * 权重（边长）



##### 2. 图的表示和分类

图的表示：**邻接矩阵**和**邻接表**

1）无向无权图

![2VQ9D.png](https://wx2.sbimg.cn/2020/07/05/2VQ9D.png)

2）有向无权图

![2VcCk.png](https://wx1.sbimg.cn/2020/07/05/2VcCk.png)

3）无向有权图

![2VyzD.png](https://wx1.sbimg.cn/2020/07/05/2VyzD.png)

4）有向有权图







#### 图的常见算法

1. DFS 模板  

![2F2zK.png](https://wx2.sbimg.cn/2020/07/05/2F2zK.png)

> 图的DFS一定要写 visted = set() 集合, 因为图里面可能会有环路

2. BFS 模板

![2FCOG.png](https://wx1.sbimg.cn/2020/07/05/2FCOG.png)

> 图的BFS也一定要写 visted = set() 集合, 因为图里面可能会有环路



#### 图的高级算法

* [连通图的个数](https://leetcode-cn.com/problems/number-of-islands/)
* [拓扑排序（Topological Sorting）](https://zhuanlan.zhihu.com/p/34871092)
* [最短路径（Shortest Path）：Dijkstra](https://www.bilibili.com/video/av25829980?from=search&seid=13391343514095937158)
* [最小生成树（Minimum Spanning Tree）](https://www.bilibili.com/video/av84820276?from=search&seid=17476598104352152051)