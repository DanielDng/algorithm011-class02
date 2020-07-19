int findMin(vector<int>& nums) {
	int left = 0, right = nums.size() - 1, mid;
	while (left < right) {
		mid = left + (right - left) / 2;
		if (nums[mid] > nums[right]) { // mid ÔÚ×ó°ë¶Î
			left = mid + 1;
		} else { // mid ÔÚÓÒ°ë¶Î
			right = mid;                
		}
	}
	return nums[left];
}