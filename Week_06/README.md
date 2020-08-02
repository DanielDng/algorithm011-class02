## 动态规划

### 递归+分治

#### 递归模板

```cpp
void recursion(int level, int param) { 
	// recursion terminator
	if (level > MAX_LEVEL) { 
		// process result 
		return ; 
  }

	// process current logic 
	process(level, param);

	// drill down 
	recursion(level + 1, param);

	// reverse the current level status if needed
}
```

 

#### 分治模板

```cpp
int divide_conquer(Problem *problem, int params) {
	// recursion terminator
	if (problem == nullptr) {
        process_result
        return return_result;
	}

	// process current problem
	subproblems = split_problem(problem, data)
	subresult1 = divide_conquer(subproblem[0], p1)
	subresult2 = divide_conquer(subproblem[1], p1)
	subresult3 = divide_conquer(subproblem[2], p1)
	...

	// merge
	result = process_result(subresult1, subresult2, subresult3)

	// revert the current level status
        
	return 0;
}
```



#### 感悟

> 面试的题目一看就很复杂，肯定可以拆分成可重复的子问题。
>
> 1. 人肉递归是低效的，很累。
> 2. 找到最近最简的方法，将其拆解成可重复解决的问题
> 3. 数学归纳法思想，抵制人肉递归的诱惑

**本质**：寻找重复性。

因为计算机的指令集（`if-else`、`for loop`、递归）具有重复性。

**人肉递归**：

如算`Fib(6)`：

* 画状态树（节点扩散是指数级的，状态是指数个，计算复杂度也是指数级（每个节点都要计算一次））
* 找重复子状态。



### 动态规划 Dynamic Programming

1. [Wiki定义](https://en.wikipedia.org/wiki/Dynamic_programming)
2. "Simplifying a complicated problem by breaking it down into simpler subproblems" (in a recursive manner)
3. Divide & Conquer + Optimal substructure (分治 + 最优子结构)



#### 关键点

动态规划 和 递归 或者 分治  没有本质上的区别，关键是看有无最优的子结构

> 若没有最优子结构，说明所有的子问题都需要计算一次，最后把所有的子问题合并在一起，也就是分治。 

**共性**：找到重复的子问题

**差异性**：DP有最优子结构、DP中途可以淘汰次优解（如果不进行淘汰的话，就很容易出现指数级的时间复杂度；进行淘汰可以将时间复杂度降低到 $O(n^2)$ 或者是 $O(n)$）





## 动态规划例题讲解

#### [509. 斐波那契数](https://leetcode-cn.com/problems/fibonacci-number/)

傻递归：

```cpp
int fib(int n) {
    if (n <= 1) return n;
    else return fib(n - 1) + fib(n - 2);
}

// Call tree has n levels
level1: 1 node
level2: 2 nodes
level3: 4 nodes

1 * 2 * 2 * 2 * ... = O(2^n)
```



简化后的代码：$O(n)$

```cpp
class Solution {
public:
    int fib(int n) {
        if (n <= 1) return n;
        vector<int> mono(n + 1, 0);
        return helper(n, mono);
    }

    int helper(int n, vector<int>& mono) {
        if (n <= 1) return n;
        if (mono[n] == 0)
            mono[n] = helper(n - 1, mono) + helper(n - 2, mono);
        return mono[n];
    }
};
```



Bottom up 自底向上（最常用的动态规划思想，一般我们就直接自底向上来循环）：

> 对于递归，尝试将其转换为for循环，自底向上进行搜索

```cpp 
int fib(int n) {
    if (n <= 1) return n;
    vector<int> a(n + 1); // 不是n，n会越界
    a[0] = 0;
    a[1] = 1;
    for (int i = 2; i <= n; ++i) {
        a[i] = a[i - 1] + a[i - 2];
    }
    return a[n];
}
```

> 注意数组越界的问题



观察，dp[n] 只与它之前的两个状态有关
上面的DP可以进一步优化,把空间复杂度降为 O(1)

```cpp
int fib(int N) {
    if (N <= 1) return N;
    int pre = 0;
    int cur = 1;
    int res;
    for (int i = 2; i <= N; ++i) {
        res = pre + cur;
        pre = cur;
        cur = res;
    }
    return res;
}
```



#### [62. 不同路径](https://leetcode-cn.com/problems/unique-paths/)

动态规划：

```cpp
int uniquePaths(int m, int n) {
    if (m == 0 || n == 0) return 0;
    if (m == 1 || n == 1) return 1;
    vector<vector<int>> dp(m, vector<int>(n, 0));
    
    // base case
    for (int col = 0; col < n; col++) {
        dp[m - 1][col] = 1;
    }
    for (int row = 0; row < m; row++) {
        dp[row][n - 1] = 1;
    }
    
    for (int row = m - 2; row >= 0; row--) {
        for (int col = n - 2; col >= 0; col --) {
            dp[row][col] = dp[row + 1][col] + dp[row][col + 1];
        }
    }
    return dp[0][0];
}
```



优化存储空间，改为用一维数据进行存储：

```cpp
int uniquePaths(int m, int n) {
    if (m == 0 || n == 0) return 0;
    if (m == 1 || n == 1) return 1;
    
    // base case,首先将最后一行的元素存储为1
    vector<int> dp(n, 1);
    for (int row = m - 2; row >= 0; row--) { // 从倒数第二行开始进行更新
        for (int col = n - 2; col >= 0; col--) { // 最后一列不更新，因为一直是1
            dp[col] = dp[col] + dp[col + 1];
        }
    }
    return dp[0];
}
```





#### [63. 不同路径 II](https://leetcode-cn.com/problems/unique-paths-ii/)

动态规划：

```cpp
int uniquePathsWithObstacles(vector<vector<int>>& obstacleGrid) {
    int m = obstacleGrid.size();
    int n = obstacleGrid[0].size();
    if (obstacleGrid[m - 1][n - 1] == 1) return 0; // 最后一个位置有障碍物时

    vector<vector<long>> dp(m, vector<long>(n)); // 计算结果数目太大，超出了int的数值范围，应改为long型
    
    // base case
    dp[m - 1][n - 1] = 1;
    for (int row = m - 2; row >= 0; row--) {
        if (obstacleGrid[row][n - 1] == 1) dp[row][n - 1] = 0;
        else dp[row][n - 1] = dp[row + 1][n - 1];
    }
    for (int col = n - 2; col >= 0; col--) {
        if (obstacleGrid[m - 1][col] == 1) dp[m - 1][col] = 0;
        else dp[m - 1][col] = dp[m - 1][col + 1];
    }

    // bottom up, 列状态转移方程
    for (int row = m - 2; row >= 0; row--) {
        for (int col = n - 2; col >= 0; col--) {
            if (obstacleGrid[row][col] == 1) dp[row][col] = 0;
            else dp[row][col] = dp[row + 1][col] + dp[row][col + 1];
        }
    }
    return dp[0][0];
}
```



####  动态规划的关键点

1. 最优子结构 `opt[n] = best_of(opt[n - 1], opt[n - 2], ...)`
2. 开个数组来储存中间状态：`opt[i]`
3. 递推公式（状态转移方程 或 DP方程）
   一维Fib：`opt[i] = opt[n - 1] + opt[n - 2]`
   二维路径： `opt[i, j] = opt[i + 1, j] + opt[i, j + 1]`（且判断 `a[i][j]`是否为空地）



#### [120. 三角形最小路径和](https://leetcode-cn.com/problems/triangle/)

**动态规划解题思路**

1. 找重复性（分治）: `problem(i, j) = min(sub(i + 1, j), sub(i + 1,j + 1)) + a[i][j]`
2. 定义状态数组:` f[i][j]`
3. 状态转移方程: `f[i][j] = min(f[i + 1][j], f[i + 1][j + 1]) + a[i][j]`



```cpp
int minimumTotal(vector<vector<int>>& triangle) {
    int n = triangle.size(); // 三角形的行数
    int max_row_length = triangle[n - 1].size(); // 三角形最后一行的长度
    // for (int i = 0; i < n; i++) {
    //     max_row_length = max(max_row_length, triangle[i].size());
    // }
    vector<vector<int>> dp(n, vector<int>(max_row_length, 0));
    
    // base case, 即三角形最后一行的元素值
    for (int i = 0; i < max_row_length; i++) {
        dp[n - 1][i] = triangle[n - 1][i];
    }
    
    // 状态转移方程
    for (int row = n - 2; row >= 0; row--) {
        # for (int col = 0; col < triangle[row].size(); col++) 也可以
        for (int col = triangle[row].size() - 1; col >= 0; col--) {
            dp[row][col] = triangle[row][col] + min(dp[row + 1][col], dp[row + 1][col + 1]);
        }
    }
    return dp[0][0];
}
```



优化存储空间，改为用一维数据进行存储：

```cpp
int minimumTotal(vector<vector<int>>& triangle) {
    int n = triangle.size();
    int max_row_length = triangle[n - 1].size();

    vector<int> dp(max_row_length, 0);
    for (int i = 0; i < max_row_length; i++) {
        dp[i] = triangle[n - 1][i];
    }
    # vector<int> dp = triangle.back(); 这样可以直接赋值

    for (int row = n - 2; row >= 0; row--) {
        # for (int col = triangle[row].size() - 1; col >= 0; col--) 是错误的，因为先更新dp[col + 1]，影响了前面dp[col]的更新
        for (int col = 0; col < triangle[row].size(); col++) {
            dp[col] = triangle[row][col] + min(dp[col], dp[col + 1]);
            
        }
    }
    return dp[0];
}
```

> `vector.back( )` : 获取 `vector` 中的最后一个元素



也可不用额外的存储空间，直接在原二维vector上面改：

```cpp
int minimumTotal(vector<vector<int>>& triangle) {
    for (int row = triangle.size() - 2; row >= 0; row--) {
        for (int col = 0; col <= row; col++) {
            triangle[row][col] += min(triangle[row + 1][col], triangle[row + 1][col + 1]);
        }
    }
    return triangle[0][0];
}
```



#### [53. 最大子序和](https://leetcode-cn.com/problems/maximum-subarray/)

动态规划，公式为：

`dp[i] = max(nums[i], nums[i] + dp[i - 1]);`

最大子序和 = 当前元素自身 最大，或者 当前元素与之前的计算结果相加之后 最大

```cpp
int maxSubArray(vector<int>& nums) {
	int n = nums.size();
    vector<int> dp(nums); // 用nums对dp进行初始化

    for (int i = 1; i < n; i++) {
        dp[i] = max(dp[i - 1] + nums[i], nums[i]) ;
    }
    
    int max_sum = INT_MIN;
    for (int item : dp) {
        max_sum = max(max_sum, item);
    }    
    return max_sum;
}
```



对上面的代码的两个for循环进行合并，简化后的代码：

```cpp
int maxSubArray(vector<int>& nums) {
    int n = nums.size();
    vector<int> dp(nums);

    int max_sum = dp[0]; // 下面的循环dp[0]没有计算在内，所以初始化为max_sum = INT_MIN会出错
    for (int i = 1; i < n; i++) {
        dp[i] = max(dp[i - 1] + nums[i], nums[i]);
        max_sum = max(max_sum, dp[i]);
    }
    return max_sum;
}
```



上面的 `dp` 数组只用到了 `dp[i]` 与 `dp[i - 1]`，所以不用开辟数组，只定义一个 `int` 类型即可。

优化存储空间：

```cpp
int maxSubArray(vector<int>& nums) {
    int n = (int)nums.size();
    if (n == 0) return 0;
    // base case
    int dp = nums[0];
    int res_sum = nums[0];
    for (int i = 1; i < n; i++){
        dp = max(dp + nums[i], nums[i]);
        res_sum = max(res_sum, dp);
    }
    return res_sum;
}
```



#### [152. 乘积最大子数组](https://leetcode-cn.com/problems/maximum-product-subarray/)

因为乘积有正有负，所以需要额外存储两个 `dp` 数组（这里直接优化为两个 `int` 型变量），即 `max_dp`，`min_dp`，分别用来存储存放的最大值和最小值。

* 当 `nums[i]` 为正数时，需要将 `max_dp` 和 `nums[i]` 进行相乘；
  公式为：`max_dp = max(max_dp * nums[i], nums[i]);`
* 当 `nums[i]` 为负数时，需要将 `min_dp` 和 `nums[i]` 进行相乘；
  公式为：`min_dp = min(min_dp * nums[i], nums[i]);`
* 最后取最大的乘积。
  `res = max(res, max_dp);`

```cpp
int maxProduct(vector<int>& nums) {
    int n = nums.size();

    int max_dp = nums[0];
    int min_dp = nums[0];
    int res_sum = nums[0];
    for (int i  = 1; i < n; ++i) {
        if (nums[i] < 0) { // 如果nums[i] < 0, 则交换最大值和最小值，因为 最小值 * 负数 变为最大值
            int temp = max_dp;
            max_dp = min_dp;
            min_dp = temp;
        }
        // 当前的元素 nums[i] 乘以前面计算的结果 或者 当前元素不乘 nums[i]
        max_dp = max(max_dp * nums[i], nums[i]); // 记录最大值
        min_dp = min(min_dp * nums[i], nums[i]); // 记录最小值
        res_sum = max(res_sum, max_dp);
    }
    return res_sum;
}
```



#### [322. 零钱兑换](https://leetcode-cn.com/problems/coin-change/)

动态规划：

状态转移方程为：`dp[i] = min(dp[i], dp[i - coins[j]] + 1)` 即使用了一枚 `coins[j]` 硬币

```cpp
int coinChange(vector<int>& coins, int amount) {
    int n = coins.size();
    vector<int> dp(amount + 1, amount + 1); // 相当于赋了一个最大值
    dp[0] = 0; // 当 amount == 0 时，结果为 0
    for (int i = 1; i <= amount; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i < coins[j]) continue; // 当前的零钱超出了总数，跳过
            dp[i] = min(dp[i], dp[i - coins[j]] + 1);
        }
    }
    return dp[amount] == amount + 1 ? -1 : dp[amount]; // dp[amount] == 最大值，说明没有更新，未找到解决方案，返回 -1
}
```



#### [198. 打家劫舍](https://leetcode-cn.com/problems/house-robber/)

方法1：

动态规划，使用**二维数组**来定义DP数组

* `dp[i][0,1]` 表示 `0..i` 天能够被偷的最高金额，`0` 表示第 `i` 个元素没有被偷，`1` 表示被偷

状态转移方程如下:

* `dp[i][0] = max(dp[i-1][0], dp[i-1][1])` ，即第 `i-1` 可以偷，可以不偷，取最大

* `dp[i][1] = dp[i-1][0] + nums[i]`

```cpp
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    vector<vector<int>> dp(n, vector<int>(2));
    // base case
    dp[0][0] = 0;
    dp[0][1] = nums[0];

    for (int i = 1; i < n; i++) {
        dp[i][0] = max(dp[i-1][0], dp[i-1][1]);
        dp[i][1] = dp[i-1][0] + nums[i];
    }
    return max(dp[n-1][0], dp[n-1][1]);
}
```



方法2：

动态规划，使用**一维数组**来定义DP数组

* `dp[i]` 表示 `0..i` 天，能够被偷的最高金额。
* 最后结果即求 `max(dp[i])`

状态转移方程：

* `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`

```cpp
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];

    vector<int> dp(n);
    dp[0] = nums[0];
    dp[1] = max(nums[0], nums[1]);
    int res = max(dp[0], dp[1]);
    for (int i = 2; i < n; ++i) {
        dp[i] = max(dp[i-1], dp[i-2] + nums[i]);
        res = max(res, dp[i]);
    }
    return res;
}
```



存储空间优化：

考虑到上式只用到了 `dp[i-1]` 和 `dp[i-2]`，因此可以只设置两个 `int` 变量互相递推就可以。

```cpp
int rob(vector<int>& nums) {
    int n = nums.size();
    if (n == 0) return 0;
    if (n == 1) return nums[0];

    int pre = nums[0];
    int cur = max(nums[0], nums[1]);

    for (int i = 2; i < n; ++i) { // 从下标2开始遍历
        int temp = cur;
        cur = max(cur, pre + nums[i]);
        pre = temp;
    }
    return cur;
}
```



 初始化时，我们选择 `f(-1) = f(0) = 0`，可以极大的简化代码

 所求的答案为 `f(n)`：

```cpp
int rob(vector<int>& nums) {
    int preMax = 0;
    int curMax = 0;

    for (int num : nums) { // 从下标0开始遍历
        int temp = curMax;
        curMax = max(curMax, preMax + num);
        preMax = temp;
    }
    return curMax;
}
```

