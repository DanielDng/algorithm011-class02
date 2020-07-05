class Solution {
public:
	vector<vector<string>> groupAnagrams(vector<string>& strs) { // ʹ�� unordered_map
		unordered_map<string, vector<string>> mp;
		for (string s : strs) {
			string t = s;
			sort(t.begin(), t.end());
			mp[t].push_back(s); // �ڶ�ά������������ .push_back()
		}
		vector<vector<string>> anagrams; // �ڶ�ά���ַ���
		for (auto p : mp) {
			anagrams.push_back(p.second);
		}
		return anagrams;
	}
}��


// ʹ��������м�������
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, vector<string>> mp;
        for (string s : strs) {
            mp[strSort(s)].push_back(s); // ���ü�������� s ���򣬷����ź���Ľ�� strSort(s)
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
            // string(int n,char c);  �� n ���ַ� c ��ʼ��
            t += string(counter[c], c + 'a'); 
        }
        return t;
    }
};