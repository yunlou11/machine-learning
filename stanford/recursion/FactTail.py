# -- coding: utf-8 --


def fact_tail(n, res):
    if n == 1:
        return res
    else:
        return fact_tail(n - 1, n * res)


def main():
    print "factorial recursion"
    print fact_tail(5, 1)
if __name__ == '__main__':
    main()