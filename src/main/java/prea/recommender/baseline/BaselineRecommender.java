package prea.recommender.baseline;

import prea.data.structure.SparseMatrix;
import prea.recommender.Recommender;
import prea.util.EvaluationMetrics;

/**
 * This is an abstract class implementing five baselines, including constant model,
 * overall average, user average, item average, and random.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public abstract class BaselineRecommender implements Recommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	protected SparseMatrix rateMatrix;
	
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public double minValue;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a constant model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 */
	public BaselineRecommender(int uc, int ic, double max, double min) {
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Build a model with given training set.
	 * 
	 * @param rm Training data set.
	 */
	@Override
	public void buildModel(SparseMatrix rm) {
		rateMatrix = rm;
	}
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Predict a rating for the given user and item.
	 * 
	 * @param userId The target user.
	 * @param itemId The target item.
	 * 
	 * @return predicted rating.
	 */
	abstract double predict(int userId, int itemId);
	
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param testMatrix The matrix with test data points used for evaluation.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	@Override
	public EvaluationMetrics evaluate(SparseMatrix testMatrix) {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		
		try {
			for (int u = 1; u <= userCount; u++) {
				int[] testItems = testMatrix.getRowRef(u).indexList();

				if (testItems != null) {
					for (int t = 0; t < testItems.length; t++) {
						int i = testItems[t];
						double prediction = predict(u, i);
						predicted.setValue(u, i, prediction);
					}
				}
			}
		}
		catch(Exception e) {
			
		}
		
		return new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
	}
}
