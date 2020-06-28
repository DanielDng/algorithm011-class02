/**
* 3.Stack：维护一个单调递减栈
* 栈内元素单调递减说明栈内每一个元素的左边界都被栈内前一个元素界定
* 当当前元素大于栈顶元素且栈非空时，入栈，表明栈顶元素被栈内的前一个元素和当前元素界定
* 计算当前元素和前一个元素之间的距离 distance
* 计算当前元素所能接水的面积，加到 ans
*/
class Solution {
public:
    int trap(vector<int>& height) {
        stack<int> mono;
        int ans = 0, current = 0;
        while (current < height.size()) {
            while (!mono.empty() && height[current] > height[mono.top()]) {
                int topValue = mono.top();
                mono.pop();
                if (mono.empty()) {
                    break;
                }
                int distance = current - mono.top() - 1;
                int boundedHeight = min(height[mono.top()], height[current]) - height[topValue];
                ans += distance * boundedHeight;
            }
            mono.push(current++);
        }
        return ans;
    }
};