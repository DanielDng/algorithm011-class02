int maxProfit(vector<int>& prices) {
	int res = 0;
	int n = prices.size();
	for (int i = 1; i < n; ++i) {
		res += max(prices[i] - prices[i - 1], 0); // 只加正数
	}
	return res;
}