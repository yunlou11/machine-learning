# -- coding: utf-8 --


class MergeSort:
    def __init__(self):
        pass

    def sort(self, array):
        length = len(array)
        if length < 2:
            return array
        left = length / 2
        right = length / 2
        return self.merge(self.sort(array[0:left]), self.sort(array[right:length]))

    @staticmethod
    def merge(left_array, right_array):
        sort_array = []
        left_index = 0
        right_index = 0
        while left_index < len(left_array) and right_index < len(right_array):
            print "left: %s right: %s" % (left_index, right_index)
            print "left array:", left_array, "right_array:", right_array
            if left_array[left_index] <= right_array[right_index]:
                sort_array.append(left_array[left_index])
                left_index += 1
            else:
                sort_array.append(right_array[right_index])
                right_index += 1
        while left_index < len(left_array):
            sort_array.append(left_array[left_index])
            left_index += 1
        while right_index < len(right_array):
            sort_array.append(right_array[right_index])
            right_index += 1
        print "sort_array", sort_array
        return sort_array


def main():
    print "merge sort"
    merge_sort = MergeSort()
    data = [5, 3, 1, 9, 10, 6, 5, 7]
    print merge_sort.sort(data)

if __name__ == '__main__':
    main()