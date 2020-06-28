# 第一周学习总结

[TOC]

### Ⅰ如何精通一个领域

1. 切碎知识点 - Chunk it up  

   * 庖丁解牛
   * 脉络相连

2. 刻意练习 - Deliberate Practicing

   * 五毒神掌，每个题过5遍
   * 针对薄弱、缺陷

3. 反馈 - Feed Back

   * 主动反馈 - 自己找

     1）看高手的代码，如Github，LeetCode

     2）第一视角直播视频

   * 被动反馈 - 高手指点

     1）Code Review

     2）教练看你打，给你反馈



### Ⅱ 刷题技巧

#### 切题四件套

1. Clarification - 和面试官多沟通，把题目看明白
2. Possible Solutions - 想所有可能的解法
   * Compare - time/space
   * Optimal - 加强优化，找到最优解
3. Coding - 多写
4. Test Cases



#### 五毒神掌

1. 第一遍
   * 5分钟 - 读题 + 思考
   * 直接看解法 - 注意多解法，多比较
   * 背诵，默写
2. 第二遍
   * 自己写，LeetCode 提交
   * 多种方法比较、体会、优化（重要的是执行时间的比较）
3. 第三遍
   * 过一天后，重复练习
4. 第四遍
   * 过一周后，重复练习
5. 第五遍
   * 面试前一周，恢复性训练



### III 数据结构总览

#### 一维

* 基础

	1. 数组 - array / string
	2. 链表 - linked list
* 高级  
  1. 栈 - stack
  2. 队列 - queue
  3. 双端队列 - deque
  4. 集合 - set
  5. 映射 - map (map / hash)



#### 二维

* 基础
  1. 树 - tree
  2. 图 - graph
* 高级
  1. 二叉搜索树 - binary search tree (red-back tree, AVL)
  2. 堆 - heap
  3. 并查集 - disjoint set
  4. 字典树 - Trie
* 特殊
  1. 位运算 - Bitwise
  2. 布隆过滤器 - BloomFilter
  3. LRU Cache



### IV 算法总览

* Branch: if-else, switch
* Iteration: for, while loop
* 递归 - Recursion (Divide & Conquer, Backtrace)
* 搜索 - Search
  * 深度优先搜索：Depth first search
  * 广度优先搜索：
    1. Breath first search
    2. A*
* 动态规划 - Dynamic Programming
* 二分查找 - Binary Search
* 贪心 - Greedy
* 数学 - Math，几何 - Geometry



### V 算法复杂度

#### 时间复杂度

* 表示 - Big O notation，只看最高复杂度
  1. O(1): Constant Complexity 常数复杂度
  2. O($log n$): Logarithmic Complexity 对数复杂度
  3. O($n$): Linear Complexity 线性时间复杂度
  4. O($n^2$): N square Complexity 平方时间复杂度
  5. O($n^3$): N cubic Complexity 立方时间复杂度
  6. O($2^n$): Exponential Growth 指数时间复杂度
  7. O($n!$): Factorial 阶乘
  
* 主定理
  [![2hegk.png](https://wx1.sbimg.cn/2020/06/28/2hegk.png)](https://sbimg.cn/image/2hegk)

    > 主定理：用于解决递归函数的时间复杂度
    >
    > （1）二分查找：在有序的数列中找到目标数，每次都是一分为二，只查一边查找下去，所以时间复杂度是 O($log n$)。
    >
    > （2）二叉树的遍历：每次一分为二，但每边都是相等的时间复杂度遍历下去。简化的思考方式：每个节点的访问且仅访问一次，所以时间复杂度为 O($n$)。
    >
    > （3）排好序的二维矩阵的二分查找：一维的数据进行二分查找是 O($log n$)，二维的矩阵进行二分查找是 O($n$)。
    >
    > （4）归并排序：所有的排序最优的方法都是 O($nlog n$)。

  

  1. 二叉树的遍历 - 前序、中序、后序：O($n$)

     > 因为每个节点访问且仅访问一次，所以时间复杂度线性于二叉树的节点总数 n。

  2. 图的遍历：O($n$)

  3. 搜索算法 - DFS、BFS：O($n$)

     > 因为搜索空间里的节点访问且仅访问一次

  4. 二分查找：O($log n$)



#### 空间复杂度

* 数组的长度

  > 此时，数组的长度就是空间复杂度

* 递归的深度

  > 递归最深的深度就是空间复杂度的最大值,如: LeetCode 70. 爬楼梯问题



### VI 数组 Array

* 元素类型：多样化（泛型：可以放任何类型的数据）

* 底层实现：内存管理器（申请数组时在内存中开辟一段连续的地址，每个地址可以直接通过内存管理器进行访问）

* 时间复杂度

  1. Prepend（从头结点增加元素）- O(1)

     > 注意：正常情况下，数组的 prepend 操作的时间复杂度是 O(n)，但是可以进行特殊处理优化到 O(1)。采用的方式是申请稍大一些的内存空间，然后在数组最开始预留一部分空间，然后 prepend 操作则是把头下标前移一个位置即可。

  2. Append（从尾结点增加元素）- O(1)

  3. Look up（随机访问某个元素）- O(1)

  4. Insert - O($n$)

  5. Delete - O($n$)



### Ⅶ 链表 Linked List

* 元素类型：一般用 class 类定义（即 Node）
* 链表类型
  1. 单链表：含 next 指针
  2. 双向链表（Double Linked List）：含 next 指针 和 prev 或 previous 指针
  3. 循环链表：尾指针 tail 指向头结点
* 时间复杂度
  1. Prepend（从头结点增加元素）- O(1)
  2. Append（从尾结点增加元素）- O(1)
  3. Look up（随机访问某个元素）- O($n$)
  4. Insert - O(1)
  5. Delete - O(1)



### Ⅷ 跳表 Skip List (针对链表在有序的情况下的优化)

> 只能用于元素有序的情况。
>
> 跳表对标的是平衡二叉树（AVL Tree）和二分查找。
>
> 现实中，因为元素的增加删除，导致元素的索引不是特别的工整。
>
> 维护成本较高，增加、删除元素要更新索引。所以增加、删除的复杂度也变成 O($log n$)了

* 特点

  1. 只能用于元素有序的情况
  2. 插入/删除/搜索 - O($log n$)
  3. 原理简单、容易实现、方便拓展、效率高
  4. 热门项目中常用来替代平衡树，如 Redis、LevelDB

* 如何对普通链表进行加速：升维（即牺牲空间来换取时间）、增加多级索引

* 时间复杂度

  1. 查询 - O(log n)

     > 若原始链表的结点的个数是 $n$，
     > 则第一级索引结点的个数是 $n/2$,
     > 第二级索引结点的个数是 $n/4$, 
     > ...
     > 第 k 级索引结点的个数是 $n/(2^k)$
     >
     > 假设索引有 h 级，最高级的索引有2个结点。
     > $n/(2^h) = 2$，
     > 从而求得 $h = log_2(n) - 1$

  2. 增加 - O($log n$)

  3. 删除 - O($log n$)
  
* 空间复杂度 - O($n$)
  
  > 假设原始链表大小为 n，每 2 个结点抽 1 个，每层结点的索引数：
  > n/2, n/4, n/8, ..., 8, 4, 2
  >
  > 假设原始链表大小为 n，每 3 个结点抽 1 个，每层结点的索引数：
  > n/3, n/9, n/27, ..., 9, 3, 1
  >
  > 因为它们加起来是收敛的。
  
  

### IX 工程中的应用

* LRU Cache - Linked List
* Redis - Skip List



### X 栈 Stack

* 先入后出 (FILO): First in - Last out
* 添加、删除 - O(1)
* 查询操作 - O($n$)



### XI 队列 Queue

* 先入先出 (FIFO): First in - First out
* 添加、删除 - O(1)
* 查询操作 - O($n$)



### XII 双端队列 Deque (Double-End Queue)

* 两端都可以 push、pop
* 添加、删除 - O(1)
* 查询操作 - O($n$)



### XIII 优先队列 Priority Queue

* 插入操作 - O(1)
* 取出操作 - O($log n$) ，按照元素的优先级取出
* 底层具体实现的数据结构较为多样和复杂：heap（堆）、BST（Binary Search Tree）、Treap





   