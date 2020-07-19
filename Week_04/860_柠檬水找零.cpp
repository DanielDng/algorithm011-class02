bool lemonadeChange(vector<int>& bills) {
	int n = bills.size();
	if (n == 0) return true;
	// 可以直接设置成如下形式
	// int c5 = 0, c10 = 0;
	unordered_map<int, int> map;
	for (int i = 0; i < n; ++i) {
		if (bills[i] == 5) {
			map[5]++;
		} else if (bills[i] == 10) {
			map[10]++;
			if (map[5] <= 0) return false;
			map[5]--;
		} else {
			if (map[10] >= 1 && map[5] >= 1) {
				map[10]--;
				map[5]--;
			} 
			else if (map[5] >= 3) map[5] -= 3;
			else return false;
		}
	}
	return true;
}