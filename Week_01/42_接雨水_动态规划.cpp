/**
* 2.动态规划：提前存储每个柱子的最大左边界以及最大右边界
* 需要两个数组 maxLeft 和 maxRight 分别提前扫描并存储相应的值
* 时间复杂度 O(n), 空间复杂度 O(n)
*/

class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        if (n == 0) {
            return 0;
        }
        vector<int> maxLeft(n), maxRight(n);
        int ans = 0;

        maxLeft[0] = height[0];
        for (int i = 1; i < n; ++i) { // 扫描最大左边界
            maxLeft[i] = max(height[i], maxLeft[i - 1]);
        }

        maxRight[n - 1] = height[n - 1];
        for (int i = n - 2; i >= 0; --i) {
            maxRight[i] = max(height[i], maxRight[i + 1]);
        }

        for (int i = 0; i < n; ++i) {
            ans += min(maxLeft[i], maxRight[i]) - height[i];
        }

        return ans;
    }
};