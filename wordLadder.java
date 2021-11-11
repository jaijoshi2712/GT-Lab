import java.util.*;

public class wordLadder {
    public int ladderLength(String beginWord, String endWord, List<String> wordList) 
    {
        HashSet<String> set = new HashSet<>();
        for(String s : wordList)
        {
            set.add(s);
        }        

        Queue<pair> queue = new LinkedList<>();
        queue.add(new pair(beginWord, 1));

        int ans = 0;

        while(queue.size() > 0)
        {
            pair rem = queue.remove();

            if(rem.s.equals(endWord))
            {
                ans = rem.count;
                break;
            }

            for(int i=0; i<rem.s.length(); i++)
            {
                char ch = 'a';
                for(int j=1; j<=26; j++)
                {
                    String temp = rem.s.substring(0,i) + ch + rem.s.substring(i + 1);

                    if(set.contains(temp) && !temp.equals(rem.s))
                    {
                        set.remove(temp);
                        queue.add(new pair(temp , rem.count + 1));
                    }

                    ch++;
                }
            }            
        }

        return ans;
        
    }
    
    public class pair
    {
        String s;
        int count;

        pair(String s, int count)
        {
            this.s = s;
            this.count = count;
        }
    }
}