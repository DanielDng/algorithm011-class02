int search(vector<int>& nums, int target) {
	int left = 0, right = nums.size() - 1, mid;
	while (left <= right) {
		mid = left + (right - left) / 2;
		if (nums[mid] == target) return mid;
		if (nums[mid] >= nums[left]) { // mid ÔÚ×ó°ë¶Î
			if (target >= nums[left] && target < nums[mid]) { // ÔÚ mid ×ó±ß
				right = mid - 1;
			} else {
				left = mid + 1;
			}
		} else { // mid ÔÚÓÒ°ë¶Î
			if (target <= nums[right] && target > nums[mid]) { // ÔÚ mid ×ó±ß
				left = mid + 1;
			} else {
				right = mid - 1;
			}
		}
	}
	return -1;
}