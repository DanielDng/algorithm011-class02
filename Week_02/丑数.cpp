// ��̬�滮����ָ�룩
int nthUglyNumber(int n) {
    if (n <= 0) return false;
    if (n == 1) return true; // base case
    int t2 = 0, t3 = 0, t5 = 0;
    vector<int> dp(n);
    dp[0] = 1;
    for (int i = 1; i < n; ++i) {
        dp[i] = min(dp[t2] * 2, min(dp[t3] * 3, dp[t5] * 5));
        if (dp[i] == dp[t2] * 2) t2++;
        if (dp[i] == dp[t3] * 3) t3++;
        if (dp[i] == dp[t5] * 5) t5++;
    }
    return dp[n - 1];
}


// С����
int nthUglyNumber(int n) {
    priority_queue<double, vector<double>, greater<double>> minHeap;
    // ������Ϊ int ���ͣ������ʱ���������Ϊ double ��û��
    double res = 1;
    for (int i = 1; i < n; ++i) {
        minHeap.push(res * 2);
        minHeap.push(res * 3);
        minHeap.push(res * 5);
        res = minHeap.top();
        minHeap.pop();
        // �жϵ�ǰ��С��Ԫ���Ƿ����ظ���
        while (!minHeap.empty() && res == minHeap.top()) {
            minHeap.pop();
        }
    }
    return res;
}

