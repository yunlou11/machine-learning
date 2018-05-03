# --coding: utf-8 --


class HeapSort:
    def __init__(self):
        pass

    def tree(self, array, length):
        i = (length - 1) / 2
        while i >= 0:
            left = 2 * i + 1
            if (2 * i + 2) < length:
                index = self.max_index(array, self.max_index(array, i, left), 2 * i + 2)
            else:
                index = self.max_index(array, i, left)
            self.swap(array, i, index)
            i -= 1
        print "tree:", array[0:length + 1], "|||", array[length + 1:len(array)]
        return array

    def sort(self, array):
        tail = len(array) - 1
        while tail > 0:
            self.tree(array, tail)
            self.swap(array, 0, tail)
            tail -= 1
        return array

    @staticmethod
    def max_index(array, i, j):
        if array[i] > array[j]:
            return i
        else:
            return j

    @staticmethod
    def swap(array, i, j):
        print "swap: %s - %s" % (i, j)
        tmp = array[i]
        array[i] = array[j]
        array[j] = tmp


def main():
    print "heap sort"
    data = [5, 3, 1, 9, 10, 6, 5, 7]
    heap_sort = HeapSort()
    print heap_sort.sort(data)
if __name__ == '__main__':
    main()