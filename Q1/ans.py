from collections import defaultdict
def main():

    n, k = map(int, input().split())
    arr=list( map(int, input().split()))
    d=defaultdict(int)
    distinct=0
    for i in range (k):
        if d[arr[i]]==0:
            distinct+=1
        d[arr[i]]+=1
    res=[]
    res.append(distinct)
    for i in range (k,n):
        d[arr[i-k]]-=1
        if d[arr[i-k]]==0:
            distinct-=1
        if d[arr[i]]==0:
            distinct+=1
        d[arr[i]]+=1
        res.append(distinct)
    
    for i in range (len(res)):
        print(res[i], end=" ")
        
    

if __name__ == "__main__":
    main()

