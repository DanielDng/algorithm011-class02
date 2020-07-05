class Solution {
public:
    bool isAnagram(string s, string t) { // 暴力排序
		if (s.size() != t.size()) {
			return false;
		}
		sort(s.begin(), s.end());
		sort(t.begin(), t.end());
		return s == t;
	}
	
	bool isAnagram(string s, string t) { // hash, map
		if (s.size() != t.size()) {
			return false;
		}
		int n = s.size();
		unordered_map<char, int> counts;
		for (int i = 0; i < n; ++i) {
			counts[s[i]]++;
			counts[t[i]]--;
		}

		for (auto count : counts) {
			if (count.second) { // 检索 map 中的 value 值是否有非 0 元素
				return false;
			}
		}

		return true;
	}
	
	bool isAnagram(string s, string t) { // 使用数组来模拟 map
        if (s.size() != t.size()) {
            return false;
        }
        int n = s.size();
        int counts[26] = {0};
        for (int i = 0; i < n; ++i) {
            counts[s[i] - 'a']++;
            counts[t[i] - 'a']--;
        }

        for (int i = 0; i < 26; ++i) {
            if (counts[i]) return false;
        }

        return true;
    }
	

	
};