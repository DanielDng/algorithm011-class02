class Solution {
private:
    void reverse(vector<int>& nums, int start, int end) {
        while (start < end) {
            int temp = nums[start];
            nums[start] = nums[end];
            nums[end] = temp;
            start++;
            end--;
        }
    }

public:
    void rotate(vector<int>& nums, int k) { // 三次旋转
        int n = nums.size();
        k %= n;
        reverse(nums, 0, n - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, n - 1);
        //或者
        // reverse(nums, 0, k);
        // reverse(nums, k + 1, n - 1);
        // reverse(nums, 0, n - 1);        

    }
};