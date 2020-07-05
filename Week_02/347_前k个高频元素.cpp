// 调用系统的 priority_queue 及 unordered_map
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