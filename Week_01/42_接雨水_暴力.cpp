/**
* 1.暴力解法：从左往右遍历每个柱子的最大左边界 maxLeft 及最大右边界 maxRight
*  ans += min(max_left, max_right) - height[i];
*/

class Solution {
public:
    int trap(vector<int>& height) {
        int n = height.size();
        if (n <= 2) {
            return 0;
        }
        int ans = 0;
        for (int i = 1; i < n - 1; ++i) { // 最左边的及最右边的柱子无法接水
            int maxLeft = 0, maxRight = 0;
            for (int j = i; j >= 0; --j) { // 找当前柱子 i 的最大左边界
                maxLeft = max(maxLeft, height[j]);
            }
            for (int j = i; j < n; ++j) { // 找当前柱子 i 的最大右边界
                maxRight = max(maxRight, height[j]);
            }
            ans += min(maxLeft, maxRight) - height[i]; // 减去当前柱子的高度即当前柱子的接水量
        }
        return ans;
    }
};