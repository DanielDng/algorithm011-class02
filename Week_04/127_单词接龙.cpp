// 双向BFS
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
	unordered_set<string> dict(wordList.begin(), wordList.end()), head, tail, *phead, *ptail;
	if (dict.find(endWord) == dict.end()) {
		// 处理 endWord 不在 wordList 中的特殊情况
		return 0;
	}
	head.insert(beginWord);
	tail.insert(endWord);
	int ladder = 2;
	while (!head.empty() && !tail.empty()) { // 当两个 set 都空时，才说明线索断了，所以要用 &&
		if (head.size() < tail.size()) { // 选择较短的 set 进行搜索
			phead = &head;
			ptail = &tail;
		} else {
			phead = &tail;
			ptail = &head;
		}
		unordered_set<string> temp; // 当temp最终为空时，就是线索断了
		for (auto it = phead -> begin(); it != phead -> end(); it++) {    
			string word = *it;
			for (int i = 0; i < word.size(); i++) {
				char t = word[i];
				for (int j = 0; j < 26; j++) {
					word[i] = 'a' + j;
					if (ptail -> find(word) != ptail -> end()) {
						return ladder; // 当前phead中的元素已经出现在了ptail中，即找到了
					}
					if (dict.find(word) != dict.end()) {
						temp.insert(word); // 将可能的选择 insert 到 temp
						dict.erase(word); // dict 中 erase，避免重复选择
					}
				}
				word[i] = t; // 字符复原
			}
		}
		ladder++;
		phead -> swap(temp); // 想比直接赋值，swap 是交换两个 set 的引用，更快
	}
	return 0;
}