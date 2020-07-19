vector<vector<char>> updateBoard(vector<vector<char>>& board, vector<int>& click) {
	int n = board.size(), m = board[0].size();
	int row = click[0], col = click[1];
	if (board[row][col] == 'M') { // 雷被挖出，结束，返回 board
		board[row][col] = 'X';
		return board;
	}

	vector<vector<int>> dirs = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}}; // 坐标转换的方向
	int num = 0; // 当前坐标周围雷的数量
	for (auto dir : dirs) {
		int new_row = row + dir[0];
		int new_col = col + dir[1];
		if (new_row >= 0 && new_row < n && new_col >= 0 && new_col < m && board[new_row][new_col] == 'M') num++; 
	}

	if (num > 0) {
		board[row][col] = num + '0'; // board 中每个元素都是字符
		// return board; // 如果不返回的话，下面的函数会继续执行，又修改为'B'了
	} else {
		board[row][col] = 'B'; // 周围没有雷
		for (auto dir : dirs) {
			int new_row = row + dir[0];
			int new_col = col + dir[1];
			if (new_row >= 0 && new_row < n && new_col >= 0 && new_col < m && board[new_row][new_col] == 'E') {
				vector<int> next_click = {new_row, new_col};
				updateBoard(board, next_click);
			}
		}
	}
	return board;
}