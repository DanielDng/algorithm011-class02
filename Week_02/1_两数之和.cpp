
// 使用 hash table 实现的代码
vector<int> twoSum(vector<int>& nums, int target) {
    //Key is the number and value is its index in the vector.
    unordered_map<int, int> hash;
    vector<int> result;
    for (int i = 0; i < nums.size(); ++i) {
        int numberToFind = target - nums[i];

        //if numberToFind is found in map, return them
        if (hash.find(numberToFind) != hash.end()) {
            result.push_back(hash[numberToFind]);
            result.push_back(i);
            return result;
        }

        //number was not found. Put it in the map.
        hash[nums[i]] = i;
    }
    return result;
}