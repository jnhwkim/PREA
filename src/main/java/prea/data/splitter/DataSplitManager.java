package prea.data.splitter;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;

/**
 * This class implements data split functions,
 * which are common in individual model selection methods. 
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public abstract class DataSplitManager {
	/*========================================
	 * Method Names
	 *========================================*/
	// Evaluation mode
	/** Randomly split train/test set. */
	public static final int SIMPLE_SPLIT = 901;
	/** Use predefined split file. */
	public static final int PREDEFINED_SPLIT = 902;
	/** Evaluation with K-fold cross-validation. */
	public static final int K_FOLD_CROSS_VALIDATION = 903;
	
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	protected SparseMatrix rateMatrix;
	/** Rating matrix for test items. Not allowed to refer during training and validation phase. */
	protected SparseMatrix testMatrix;
	/** The number of users. */
	protected int userCount;
	/** The number of items. */
	protected int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public int maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public int minValue;
	/** Average of ratings for each user. */
	protected static SparseVector userRateAverage;
	/** Average of ratings for each item. */
	protected static SparseVector itemRateAverage;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/** Construct a data set manager. */
	public DataSplitManager(SparseMatrix originalMatrix, int max, int min) {
		rateMatrix = originalMatrix;
		maxValue = max;
		minValue = min;
		
		int[] len = originalMatrix.length();
		userCount = len[0] - 1;
		itemCount = len[1] - 1;
		
		testMatrix = new SparseMatrix(userCount+1, itemCount+1);
		userRateAverage = new SparseVector(userCount+1);
		itemRateAverage = new SparseVector(itemCount+1);
	}
	
	/** Items in testMatrix are moved back to original rateMatrix. */
	protected void recoverTestItems() {
		for (int u = 1; u <= userCount; u++) {
			int[] itemList = testMatrix.getRowRef(u).indexList();
			
			if (itemList != null) {
				for (int i : itemList) {
					rateMatrix.setValue(u, i, testMatrix.getValue(u, i));
				}
			}
		}
		
		testMatrix = new SparseMatrix(userCount+1, itemCount+1);
	}
	
	/**
	 * Calculate average of ratings for each user and each item.
	 * Calculated results are stored in two arrays, userRateAverage and itemRateAverage.
	 * This method should be called after splitting train and test data.
	 **/
	protected void calculateAverage(double defaultValue) {
		// Calculate user Rate Average:
		for (int u = 1; u <= userCount; u++) {
			SparseVector v = rateMatrix.getRowRef(u);
			double avg = v.average();
			if (Double.isNaN(avg)) { // no rate is available: set it as median value.
				avg = defaultValue;
			}
			userRateAverage.setValue(u, avg);
		}
		
		// Calculate item Rate Average:
		for (int i = 1; i <= itemCount; i++) {
			SparseVector j = rateMatrix.getColRef(i);
			double avg = j.average();
			if (Double.isNaN(avg)) { // no rate is available: set it as median value.
				avg = defaultValue;
			}
			itemRateAverage.setValue(i, avg);
		}
	}
	
	/**
	 * Getter method for rating matrix with test data.
	 * 
	 * @return Rating matrix with test data.
	 */
	public SparseMatrix getTestMatrix() {
		return testMatrix;
	}
	
	/**
	 * Getter method for average of each user's rating.
	 * 
	 * @return A sparse vector with each user's rating.
	 */
	public SparseVector getUserRateAverage() {
		return userRateAverage;
	}
	
	/**
	 * Getter method for average of each item's rating.
	 * 
	 * @return A sparse vector with each item's rating.
	 */
	public SparseVector getItemRateAverage() {
		return itemRateAverage;
	}
}
