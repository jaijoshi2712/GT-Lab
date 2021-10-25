import os
import graph


# the tricky part of a practical problem is always connecting vertices
# we need to do a traversal on every single word in txt file
# for each word, we remove a letter and replace it by '_'
# we build up a dictionary and let words fill in the dictionary
# for instance, word 'pale' can be converted to 'p_le'
# word 'pole' can be converted to 'p_le' as well
# therefore, they should be under the same key of the dictionary
def check_around(words):
    D = {}
    for i in words:
        for j in range(4):

            # replace a letter with _
            string = i[:j] + '_' + i[j + 1:]

            # find the target words
            try:
                D[string].append(i)
            except KeyError:
                D[string] = [i]

    # every word under the same key should connect with each other
    # thats how we build up edges in a graph adt
    ADT = graph.graph()
    for k in D:
        for l in D[k]:
            for m in D[k]:
                if l != m:
                    ADT.append(l, m, 1)

    return ADT

#to simplify our problem
#lets upload a small txt file with selected words of four letters
f=open('word ladder.txt','r')
words=[]
for i in f.readlines():
    words.append(i.replace('\n',''))

ADT=check_around(words)

#bfs can find the optimal path easily
num_of_words_bfs,path_bfs=graph.bfs_path(ADT,'pale','soul')
num_of_v_bfs=len([i for i in ADT.route() if ADT.route()[i]==1])
ADT.clear(whole=True)

print(f'length of the path:',num_of_words_bfs)
print(f'number of vertices BFS has travelled:',num_of_v_bfs)


#dfs can find the path, but may not be the optimal

num_of_words_dfs,path_dfs=graph.dfs_path(ADT,'pale','soul')
num_of_v_dfs=len([i for i in ADT.route() if ADT.route()[i]==1])
ADT.clear(whole=True)

print(f'length of the path:',num_of_words_dfs)
print(f'number of vertices DFS has travelled:',num_of_v_dfs)

#dijkstra is guaranteed to find the optimal
num_of_words_dijkstra,path_dijkstra=graph.dijkstra(ADT,'pale','soul')
num_of_v_dijkstra=len([i for i in ADT.route() if ADT.route()[i]==1])
ADT.clear(whole=True)

print(f'length of the path:',num_of_words_dijkstra)
print(f'number of vertices Dijkstra has travelled:',num_of_v_dijkstra)
#ta-da
#our solution is here
#it takes six steps from pale to soul
print(path_bfs)