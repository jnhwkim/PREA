package prea.data.splitter;

import prea.data.structure.SparseMatrix;
import prea.util.Sort;

/**
 * This class implements K-fold cross-validation.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class KfoldCrossValidation extends DataSplitManager {
	private SparseMatrix assign;
	private int foldCount;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/** Construct an instance for K-fold cross-validation. */
	public KfoldCrossValidation(SparseMatrix originalMatrix, int k, int max, int min) {
		super(originalMatrix, max, min);
		divideFolds(k);
	}
	
	/**
	 * Divide the original rating matrix into k-fold.
	 * Illegal k would be adjusted automatically. 
	 * 
	 * @param k The index for desired fold.
	 */
	private void divideFolds(int k) {
		assign = new SparseMatrix(userCount+1, itemCount+1);
		int rateCount = rateMatrix.itemCount();
		
		if (k > rateCount) {
			foldCount = rateCount;
		}
		else if (k >= 2) {
			foldCount = k;
		}
		else {
			foldCount = 2;
		}
	
		double[] rdm = new double[rateCount];
		int[] fold = new int[rateCount];
		double indvCount = (double) rateCount / (double) foldCount; 
		
		for (int i = 0; i < rateCount; i++) {
			rdm[i] = Math.random();
			fold[i] = (int) (i / indvCount) + 1;
		}
		
		Sort.quickSort(rdm, fold, 0, rateCount-1, true);
		
		int f = 0;
		for (int u = 1; u <= userCount; u++) {
			int[] itemList = rateMatrix.getRowRef(u).indexList();
			if (itemList != null) {
				for (int i : itemList) {
					assign.setValue(u, i, fold[f]);
					f++;
				}
			}
		}
	}
	
	/**
	 * Return the k-th fold as test set (testMatrix),
	 * making all the others as train set in rateMatrix.
	 * 
	 * @param k The index for desired fold.
	 * @return Rating matrix with test data with data points in k-th fold.
	 */
	public SparseMatrix getKthFold(int k) {
		if (k > foldCount || k < 1) {
			return null;
		}
		else {
			recoverTestItems();
			
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = rateMatrix.getRowRef(u).indexList();
				if (itemList != null) {
					for (int i : itemList) {
						if (assign.getValue(u, i) == k) {
							testMatrix.setValue(u, i, rateMatrix.getValue(u, i));
							rateMatrix.setValue(u, i, 0.0);
						}
					}
				}
			}
			
			calculateAverage((maxValue + minValue) / 2);
			
			return testMatrix;
		}
	}
}
