class Solution {
public:
    int countSubstrings(string s) {
        int n = s.length();
        if(n == 0)
            return 0;
        int count = n;
        vector<vector<bool>> dp(n,vector<bool>(n,false));
        for(int i = 0; i < n; i++)
            dp[i][i] = true;
        for(int k = 2; k <= n; k++)
        {
            for(int i = 0; i <= n-k; i++)
            {
                int j = i+k-1;
                if(k == 2)
                    dp[i][j] = s[i] == s[j];
                else
                    dp[i][j] = dp[i+1][j-1] && s[i] == s[j];
                if(dp[i][j] == true)
                    count++;
            }
        }
        return count;
    }
};