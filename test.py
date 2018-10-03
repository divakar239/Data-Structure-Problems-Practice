def f2(a):
    a=max(a,12)
    sum+=1

def f1(a):
    sum = 0
    f2(a)
    return sum

def main():
    a=2
    f1(a)

if __name__ == '__main()__':
    main()