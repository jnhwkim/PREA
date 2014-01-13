package prea.util;

/**
 * This is a class implementing sort functions in various data type.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class Sort {
	/**
	 * Return k largest elements (sorted) and their indices from a given array.
	 * The original array will be changed, so refer to the first k element of array1 and array2 after calling this method.
	 * 
	 * @param array1 original array of data elements
	 * @param array2 original array containing data index
	 * @param first the first element in the array. Use 0 to deal with the whole array.
	 * @param last the last element in the array. Use the maximum index of the array to deal with the whole array.
	 * @param k the number of items
	 */
	public static void kLargest(double[] array1, int[] array2, int first, int last, int k) {
		int pivotIndex;
		int firstIndex = first;
		int lastIndex = last;

		while (lastIndex > k*10) {
			pivotIndex = partition(array1, array2, firstIndex, lastIndex, false);

			if (pivotIndex < k) {
				firstIndex = pivotIndex + 1;
			}
			else if (pivotIndex < k*10) { // go out and sort
				lastIndex = pivotIndex;
				break;
			}
			else {
				lastIndex = pivotIndex;
			}
		}

		quickSort(array1, array2, first, lastIndex, false);
	}

	/**
	 * Return k smallest elements (sorted) and their indices from a given array.
	 * The original array will be changed, so refer to the first k element of array1 and array2 after calling this method.
	 * 
	 * @param array1 original array of data elements
	 * @param array2 original array containing data index
	 * @param first the first element in the array. Use 0 to deal with the whole array.
	 * @param last the last element in the array. Use the maximum index of the array to deal with the whole array.
	 * @param k the number of items
	 */
	public static void kSmallest(double[] array1, int[] array2, int first, int last, int k) {
		int pivotIndex;
		int firstIndex = first;
		int lastIndex = last;

		while (lastIndex > k*10) {
			pivotIndex = partition(array1, array2, firstIndex, lastIndex, true);

			if (pivotIndex < k) {
				firstIndex = pivotIndex + 1;
			}
			else if (pivotIndex < k*10) { // go out and sort
				lastIndex = pivotIndex;
				break;
			}
			else {
				lastIndex = pivotIndex;
			}
		}

		quickSort(array1, array2, first, lastIndex, true);
	}

	/**
	 * Sort the given array. The original array will be sorted.
	 * 
	 * @param array original array of data elements
	 * @param first the first element to be sorted in the array. Use 0 for sorting the whole array.
	 * @param last the last element to be sorted in the array. Use the maximum index of the array for sorting the whole array.
	 * @param increasingOrder indicating the sort is in increasing order. Use true for increasing order, false for decreasing order.
	 */
	public static void quickSort(int[] array, int first, int last, boolean increasingOrder) {
		int pivotIndex;

		if (first < last) {
			pivotIndex = partition(array,first,last,increasingOrder);
			quickSort(array,first,pivotIndex-1,increasingOrder);
			quickSort(array,pivotIndex+1,last,increasingOrder);
		}
	}

	/**
	 * Sort the given array, and returns original index as well. The original array will be sorted.
	 * 
	 * @param array1 original array of data elements
	 * @param array2 original array containing data index
	 * @param first the first element to be sorted in the array. Use 0 for sorting the whole array.
	 * @param last the last element to be sorted in the array. Use the maximum index of the array for sorting the whole array.
	 * @param increasingOrder indicating the sort is in increasing order. Use true for increasing order, false for decreasing order.
	 */
	public static void quickSort(double[] array1, int[] array2, int first, int last, boolean increasingOrder) {
		int pivotIndex;

		if (first < last) {
			pivotIndex = partition(array1,array2,first,last,increasingOrder);
			quickSort(array1,array2,first,pivotIndex-1,increasingOrder);
			quickSort(array1,array2,pivotIndex+1,last,increasingOrder);
		}
	}

	/**
	 * Sort the given array, and returns original index as well. The original array will be sorted.
	 * 
	 * @param array1 original array of data elements of type int
	 * @param array2 original array containing data index
	 * @param first the first element to be sorted in the array. Use 0 for sorting the whole array.
	 * @param last the last element to be sorted in the array. Use the maximum index of the array for sorting the whole array.
	 * @param increasingOrder indicating the sort is in increasing order. Use true for increasing order, false for decreasing order.
	 */
	public static void quickSort(int[] array1, int[] array2, int first, int last, boolean increasingOrder) {
		int pivotIndex;

		if (first < last) {
			pivotIndex = partition(array1,array2,first,last,increasingOrder);
			quickSort(array1,array2,first,pivotIndex-1,increasingOrder);
			quickSort(array1,array2,pivotIndex+1,last,increasingOrder);
		}
	}

	/**
	 * Sort the given array, and returns original index as well. The original array will be sorted.
	 * 
	 * @param array1 original array of data elements of type int
	 * @param array2 original array containing data of type double
	 * @param first the first element to be sorted in the array. Use 0 for sorting the whole array.
	 * @param last the last element to be sorted in the array. Use the maximum index of the array for sorting the whole array.
	 * @param increasingOrder indicating the sort is in increasing order. Use true for increasing order, false for decreasing order.
	 */
	public static void quickSort(int[] array1, double[] array2, int first, int last, boolean increasingOrder) {
		int pivotIndex;

		if (first < last) {
			pivotIndex = partition(array1,array2,first,last,increasingOrder);
			quickSort(array1,array2,first,pivotIndex-1,increasingOrder);
			quickSort(array1,array2,pivotIndex+1,last,increasingOrder);
		}
	}

	/**
	 * Partition the given array into two section: smaller and larger than threshold.
	 * The threshold is selected from the first element of original array.
	 * 
	 * @param array original array of data elements
	 * @param first the first element in the array
	 * @param last the last element in the array
	 * @param increasingOrder indicating the sort is in increasing order. Use true for increasing order, false for decreasing order.
	 * @return the index of threshold item after partitioning
	 */
	private static int partition (int[] array, int first, int last, boolean increasingOrder) {
		int tmpInt;
		int pivot = array[first];

		int lastS1 = first;

		for (int firstUnknown = first+1; firstUnknown <= last; ++firstUnknown) {
			if (increasingOrder) {
				if (array[firstUnknown] < pivot) {
					++lastS1;

					tmpInt = array[firstUnknown];
					array[firstUnknown] = array[lastS1];
					array[lastS1] = tmpInt;
				}
			}
			else {
				if (array[firstUnknown] > pivot) {
					++lastS1;

					tmpInt = array[firstUnknown];
					array[firstUnknown] = array[lastS1];
					array[lastS1] = tmpInt;
				}
			}
		}

		tmpInt = array[first];
		array[first] = array[lastS1];
		array[lastS1] = tmpInt;

		return lastS1;
	}

	/**
	 * Partition the given array into two section: smaller and larger than threshold.
	 * The threshold is selected from the first element of original array.
	 * 
	 * @param array1 original array of data elements
	 * @param array2 original array containing data index
	 * @param first the first element in the array
	 * @param last the last element in the array
	 * @param increasingOrder indicating the sort is in increasing order. Use true for increasing order, false for decreasing order.
	 * @return the index of threshold item after partitioning 
	 */
	private static int partition (double[] array1, int[] array2, int first, int last, boolean increasingOrder) {
		double tmpDouble;
		int tmpInt;
		double pivot = array1[first];

		int lastS1 = first;

		for (int firstUnknown = first+1; firstUnknown <= last; ++firstUnknown) {
			if (increasingOrder) {
				if (array1[firstUnknown] < pivot) {
					++lastS1;

					tmpDouble = array1[firstUnknown];
					array1[firstUnknown] = array1[lastS1];
					array1[lastS1] = tmpDouble;

					tmpInt = array2[firstUnknown];
					array2[firstUnknown] = array2[lastS1];
					array2[lastS1] = tmpInt;
				}
			}
			else {
				if (array1[firstUnknown] > pivot) {
					++lastS1;

					tmpDouble = array1[firstUnknown];
					array1[firstUnknown] = array1[lastS1];
					array1[lastS1] = tmpDouble;

					tmpInt = array2[firstUnknown];
					array2[firstUnknown] = array2[lastS1];
					array2[lastS1] = tmpInt;
				}
			}
		}

		tmpDouble = array1[first];
		array1[first] = array1[lastS1];
		array1[lastS1] = tmpDouble;

		tmpInt = array2[first];
		array2[first] = array2[lastS1];
		array2[lastS1] = tmpInt;

		return lastS1;
	}
	/**
	 * Partition the given array into two section: smaller and larger than threshold.
	 * The threshold is selected from the first element of original array.
	 * 
	 * @param array1 original array of data elements of type int
	 * @param array2 original array containing data index
	 * @param first the first element in the array
	 * @param last the last element in the array
	 * @param increasingOrder indicating the sort is in increasing order. Use true for increasing order, false for decreasing order.
	 * @return the index of threshold item after partitioning 
	 */
	private static int partition (int[] array1, int[] array2, int first, int last, boolean increasingOrder) {
		int tmp1;
		int tmpInt;
		int pivot = array1[first];

		int lastS1 = first;

		for (int firstUnknown = first+1; firstUnknown <= last; ++firstUnknown) {
			if (increasingOrder) {
				if (array1[firstUnknown] < pivot) {
					++lastS1;

					tmp1 = array1[firstUnknown];
					array1[firstUnknown] = array1[lastS1];
					array1[lastS1] = tmp1;

					tmpInt = array2[firstUnknown];
					array2[firstUnknown] = array2[lastS1];
					array2[lastS1] = tmpInt;
				}
			}
			else {
				if (array1[firstUnknown] > pivot) {
					++lastS1;

					tmp1 = array1[firstUnknown];
					array1[firstUnknown] = array1[lastS1];
					array1[lastS1] = tmp1;

					tmpInt = array2[firstUnknown];
					array2[firstUnknown] = array2[lastS1];
					array2[lastS1] = tmpInt;
				}
			}
		}

		tmp1 = array1[first];
		array1[first] = array1[lastS1];
		array1[lastS1] = tmp1;

		tmpInt = array2[first];
		array2[first] = array2[lastS1];
		array2[lastS1] = tmpInt;

		return lastS1;
	}

	/**
	 * Partition the given array into two section: smaller and larger than threshold.
	 * The threshold is selected from the first element of original array.
	 * 
	 * @param array1 original array of data elements of type int
	 * @param array2 original array containing data of type double
	 * @param first the first element in the array
	 * @param last the last element in the array
	 * @param increasingOrder indicating the sort is in increasing order. Use true for increasing order, false for decreasing order.
	 * @return the index of threshold item after partitioning 
	 */
	private static int partition (int[] array1, double[] array2, int first, int last, boolean increasingOrder) {
		int tmp1;
		double tmp2;
		int pivot = array1[first];

		int lastS1 = first;

		for (int firstUnknown = first+1; firstUnknown <= last; ++firstUnknown) {
			if (increasingOrder) {
				if (array1[firstUnknown] < pivot) {
					++lastS1;

					tmp1 = array1[firstUnknown];
					array1[firstUnknown] = array1[lastS1];
					array1[lastS1] = tmp1;

					tmp2 = array2[firstUnknown];
					array2[firstUnknown] = array2[lastS1];
					array2[lastS1] = tmp2;
				}
			}
			else {
				if (array1[firstUnknown] > pivot) {
					++lastS1;

					tmp1 = array1[firstUnknown];
					array1[firstUnknown] = array1[lastS1];
					array1[lastS1] = tmp1;

					tmp2 = array2[firstUnknown];
					array2[firstUnknown] = array2[lastS1];
					array2[lastS1] = tmp2;
				}
			}
		}

		tmp1 = array1[first];
		array1[first] = array1[lastS1];
		array1[lastS1] = tmp1;

		tmp2 = array2[first];
		array2[first] = array2[lastS1];
		array2[lastS1] = tmp2;

		return lastS1;
	}
}
