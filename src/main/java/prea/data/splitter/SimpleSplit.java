package prea.data.splitter;

import prea.data.structure.SparseMatrix;

/**
 * This class helps to split data matrix into train set and test set,
 * based on the test set ratio defined by the user.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class SimpleSplit extends DataSplitManager {
	/*========================================
	 * Constructors
	 *========================================*/
	/** Construct an instance for simple splitter. */
	public SimpleSplit(SparseMatrix originalMatrix, double testRatio, int max, int min) {
		super(originalMatrix, max, min);
		split(testRatio);
		calculateAverage((maxValue + minValue) / 2);
	}
	
	/**
	 * Items which will be used for test purpose are moved from rateMatrix to testMatrix.
	 * 
	 * @param testRatio proportion of items which will be used for test purpose. 
	 *  
	 */
	private void split(double testRatio) {
		if (testRatio > 1 || testRatio < 0) {
			return;
		}
		else {
			recoverTestItems();
			
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = rateMatrix.getRowRef(u).indexList();
				
				if (itemList != null) {
					for (int i : itemList) {
						double rdm = Math.random();
						
						if (rdm < testRatio) {
							testMatrix.setValue(u, i, rateMatrix.getValue(u, i));
							rateMatrix.setValue(u, i, 0.0);
						}
					}
				}
			}
		}
	}
}
