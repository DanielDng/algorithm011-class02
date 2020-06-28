/**
* 4.双指针
* 只要 right_max[i] > left_max[i]，积水高度将由 left_max 决定，类似地 left_max[i] > right_max[i] 时，积水高度将有 right_max[i] 决定
* 遍历时维护两个变量 left_max 和 right_max，当 height[left] < height[right] 时，更新 left_max，当 height[right] <= height[left] 时，更新 right_max
*/

class Solution {
public:
    int trap(vector<int>& height) {
        int left_max = 0, right_max = 0;
        int ans = 0;
        int left = 0, right = height.size() - 1;

        while (left < right) {
            if (height[left] < height[right]) { // update left_max
                height[left] > left_max ? left_max = height[left] : ans += left_max - height[left];
                ++left;
            }
            else { // update right_max
                height[right] > right_max ? right_max = height[right] : ans += right_max - height[right];
                --right;
            }
        }
        return ans;
    }
};