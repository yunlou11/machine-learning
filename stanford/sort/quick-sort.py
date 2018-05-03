# --coding: utf-8 --


class QuickSort:
    def __init__(self):
        pass

    def sort(self, array, low, high):
        if low < high:
            middle_index = self.sub_sort(array, low, high)
            print array
            self.sort(array, low, middle_index - 1)
            self.sort(array, middle_index + 1, high)
        return array

    def sub_sort(self, array, low, high):
        print "sub_sort --------------------------------%s - %s" % (low, high)
        pivot = array[low]
        i = low + 1
        j = high
        while 1:
            while j > low and array[j] >= pivot:
                print "j:", low, j, array[j] >= pivot
                j -= 1
            while i <= high and array[i] < pivot:
                print "i:", low, i, array[i] < pivot
                i += 1
            if i < j:
                self.swap(array, i, j)
            else:
                break
        self.swap(array, low, j)
        return j

    @staticmethod
    def swap(array, i, j):
        print "swap: %s - %s" % (i, j)
        tmp = array[i]
        array[i] = array[j]
        array[j] = tmp


def main():
    print "quick sort"
    quick_sort = QuickSort()
    # data = [21, 32, 43, 98, 54, 45, 23, 4, 66, 86]
    data = [5, 3, 1, 9, 10, 6, 5, 7]
    print "result:  %s\n" % quick_sort.sort(data, 0, len(data) - 1)

if __name__ == '__main__':
    main()