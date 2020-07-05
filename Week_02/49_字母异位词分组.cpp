class Solution {
public:
	vector<vector<string>> groupAnagrams(vector<string>& strs) { // 使用 unordered_map
		unordered_map<string, vector<string>> mp;
		for (string s : strs) {
			string t = s;
			sort(t.begin(), t.end());
			mp[t].push_back(s); // 第二维是向量，所以 .push_back()
		}
		vector<vector<string>> anagrams; // 第二维是字符串
		for (auto p : mp) {
			anagrams.push_back(p.second);
		}
		return anagrams;
	}
}；


// 使用数组进行计数排序
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;
        for (string s : strs) {
            mp[strSort(s)].push_back(s); // 利用计数数组对 s 排序，返回排好序的结果 strSort(s)
        }

        vector<vector<string>> anagrams;
        for (auto p : mp) {
            anagrams.push_back(p.second);
        }
        return anagrams;
    }
private:
    string strSort(string s) {
        int counter[26] = {0};
        for (char c : s) {
            counter[c - 'a']++;
        }
        string t;
        for (int c = 0; c < 26; ++c) {
            // string(int n,char c);  用 n 个字符 c 初始化
            t += string(counter[c], c + 'a'); 
        }
        return t;
    }
};