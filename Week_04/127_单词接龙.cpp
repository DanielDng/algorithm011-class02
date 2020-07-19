// ˫��BFS
int ladderLength(string beginWord, string endWord, vector<string>& wordList) {
	unordered_set<string> dict(wordList.begin(), wordList.end()), head, tail, *phead, *ptail;
	if (dict.find(endWord) == dict.end()) {
		// ���� endWord ���� wordList �е��������
		return 0;
	}
	head.insert(beginWord);
	tail.insert(endWord);
	int ladder = 2;
	while (!head.empty() && !tail.empty()) { // ������ set ����ʱ����˵���������ˣ�����Ҫ�� &&
		if (head.size() < tail.size()) { // ѡ��϶̵� set ��������
			phead = &head;
			ptail = &tail;
		} else {
			phead = &tail;
			ptail = &head;
		}
		unordered_set<string> temp; // ��temp����Ϊ��ʱ��������������
		for (auto it = phead -> begin(); it != phead -> end(); it++) {    
			string word = *it;
			for (int i = 0; i < word.size(); i++) {
				char t = word[i];
				for (int j = 0; j < 26; j++) {
					word[i] = 'a' + j;
					if (ptail -> find(word) != ptail -> end()) {
						return ladder; // ��ǰphead�е�Ԫ���Ѿ���������ptail�У����ҵ���
					}
					if (dict.find(word) != dict.end()) {
						temp.insert(word); // �����ܵ�ѡ�� insert �� temp
						dict.erase(word); // dict �� erase�������ظ�ѡ��
					}
				}
				word[i] = t; // �ַ���ԭ
			}
		}
		ladder++;
		phead -> swap(temp); // ���ֱ�Ӹ�ֵ��swap �ǽ������� set �����ã�����
	}
	return 0;
}