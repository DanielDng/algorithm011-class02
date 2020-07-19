void DFSMarking(vector<vector<char>>& grid, int i, int j) {
	if (i < 0 || j < 0 || i >= grid.size() || j >= grid[0].size() || grid[i][j] == '0') 
		return ;
	grid[i][j] = '0';
	DFSMarking(grid, i + 1, j);
	DFSMarking(grid, i - 1, j);
	DFSMarking(grid, i, j + 1);
	DFSMarking(grid, i, j - 1);
}

int numIslands(vector<vector<char>>& grid) {
	int count = 0;
	for (int i = 0; i < grid.size(); ++i) {
		for (int j = 0; j < grid[i].size(); ++j) {
			if (grid[i][j] == '1') {
				count++;
				DFSMarking(grid, i, j);
			}
		}
	}
	return count;
}