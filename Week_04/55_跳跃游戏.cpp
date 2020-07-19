// ����1
bool canJump(vector<int>& nums) {
	int n = nums.size();
	int longestJump = 0;
	for (int i = 0; i < n; ++i) {
		if (i > longestJump) return false;
		longestJump = max(longestJump, nums[i] + i);
		if (longestJump >= n - 1) return true; 
	}
	return false;
}

// ����2
bool canJump(vector<int>& nums) {
	int n = nums.size();
	if (n == 0) return false;
	int endReachable = n - 1;
	for (int i = n - 1; i >= 0; --i) {
		if (nums[i] + i >= endReachable) { // nums[i] + i �е� i �����������߹��ľ��룬�ټ��� nums[i] �������ܷ�
			endReachable = i;
		}
	}
	return endReachable == 0;
}