import java.util.*;

public class knightsTour {

    public static void main(String[] args) throws Exception 
    {
        Scanner scn = new Scanner(System.in);
        
        int n = scn.nextInt();
        int r = scn.nextInt();
        int c = scn.nextInt();
        
        int[][] chess = new int[n][n];
        
        int[] helperr = {-2,-1,1,2,2,1,-1,-2};
        int[] helperc = {1,2,2,1,-1,-2,-2,-1};
        
        printKnightsTour(chess , r , c , 1, helperr, helperc);
    }
    
    public static void printKnightsTour(int[][] chess, int r, int c, int upcomingMove, int[] helperr, int[] helperc) 
    {
        if(r < 0 || c < 0 || r >= chess.length || c >= chess.length) return;
        if(chess[r][c] > 0) return;
        if(upcomingMove == chess.length * chess.length)
        {
            chess[r][c] = upcomingMove;
            displayBoard(chess);
            chess[r][c] = 0;
        }
        
        chess[r][c] = upcomingMove;
        
        for(int i=0; i<8; i++)
        {
            printKnightsTour(chess , r + helperr[i] , c + helperc[i] , upcomingMove + 1, helperr, helperc);
        }
        
        chess[r][c] = 0;
    }

    public static void displayBoard(int[][] chess){
        for(int i = 0; i < chess.length; i++){
            for(int j = 0; j < chess[0].length; j++){
                System.out.print(chess[i][j] + " ");
            }
            System.out.println();
        }

        System.out.println();
    }
}