class Solution {
private:
    vector<vector<int>> res;
    void dfs(int n, int k, int start, vector<int>& path) {
        // terminator
        if (path.size() == k) {
            res.push_back(path);
            return ;
        }
        // process
        // for (int i = start; i <= n; ++i) { 未剪枝
        for (int i = start; i <= n - (k - path.size()) + 1; ++i) { // 剪枝后
            path.push_back(i);
            // drill down
            dfs(n, k, i + 1, path);
            // reverse states
            path.pop_back();
        }
    }

public:
    vector<vector<int>> combine(int n, int k) {
        if (n <= 0 || k <= 0 || k > n) return {};
        vector<int> path;
        dfs(n, k, 1, path);
        return res;
    }
};