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