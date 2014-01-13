package prea.recommender.etc;

import prea.data.structure.DenseMatrix;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.recommender.Recommender;
import prea.util.EvaluationMetrics;

/**
 * This is a class implementing Slope-One algorithm.
 * Technical detail of the algorithm can be found in
 * Daniel Lemire and Anna Maclachlan, Slope One Predictors for Online Rating-Based Collaborative Filtering,
 * Society for Industrial Mathematics, 05:471-480, 2005.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class SlopeOne implements Recommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public double minValue;
	
	/** Prepared difference matrix */
	private DenseMatrix diffMatrix;
	/** Prepared frequency matrix */
	private DenseMatrix freqMatrix;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a Fast NPCA model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 */
	public SlopeOne(int uc, int ic, double max, double min) {
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
	}

	/*========================================
	 * Model Builder
	 *========================================*/
	/** 
	 * Build a model with the given data and algorithm.
	 * 
	 * @param rm The rating matrix with train data.
	 */
	@Override
	public void buildModel(SparseMatrix rm) {
		rateMatrix = rm;
		diffMatrix = new DenseMatrix(itemCount+1, itemCount+1);
		freqMatrix = new DenseMatrix(itemCount+1, itemCount+1);
		
		for (int u = 1; u <= userCount; u++) {
			SparseVector ratedItems = rateMatrix.getRowRef(u);
			int[] itemList = ratedItems.indexList();
			
			if (itemList != null) {
				for (int i : itemList) {
					for (int j : itemList) {
						double oldCount = freqMatrix.getValue(i, j);
						double oldDiff = diffMatrix.getValue(i, j);
						double observedDiff = rateMatrix.getValue(u, i) - rateMatrix.getValue(u, j);
						
						freqMatrix.setValue(i, j, oldCount + 1);
						diffMatrix.setValue(i, j, oldDiff + observedDiff);
					}
				}
			}
		}
		
		for (int j = 1; j <= itemCount; j++) {
			for (int i = 1; i <= itemCount; i++) {
				double count = freqMatrix.getValue(j, i);
				
				if (count > 0) {
					double oldvalue = diffMatrix.getValue(j, i);
					diffMatrix.setValue(j, i, oldvalue / count);
				}
			}
		}
	}
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @param testMatrix The rating matrix with test data.
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
					SparseVector predictedForUser = getEstimation(u, testItems);
					
					for (int t = 0; t < testItems.length; t++) {
						int i = testItems[t];
						double prediction = predictedForUser.getValue(i);
						predicted.setValue(u, i, prediction);
					}
				}
			}
		}
		catch(Exception e) {
			
		}
		
		return new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
	}
	
	/**
	 * Estimate of ratings for a given user and a set of test items.
	 * 
	 * @param u The user number.
	 * @param testItems The list of items to be predicted.
	 * 
	 * @return A list containing predicted rating scores.
	 */
	private SparseVector getEstimation(int u, int[] testItems) {
		SparseVector result = new SparseVector(itemCount+1);
		SparseVector predictions = new SparseVector(itemCount+1);
		SparseVector frequencies = new SparseVector(itemCount+1);
		
		int[] ratedItems = rateMatrix.getRowRef(u).indexList();
		
		if (ratedItems == null) {
			for (int t = 0; t < testItems.length; t++) {
				result.setValue(testItems[t], (maxValue + minValue) / 2);
			}
		}
		else {
			for (int i : ratedItems) {
				for (int j : testItems) {
					double newValue = (diffMatrix.getValue(j, i) + rateMatrix.getValue(u, i)) * freqMatrix.getValue(j, i);
			        predictions.setValue(j, predictions.getValue(j) + newValue);
			        frequencies.setValue(j, frequencies.getValue(j) + freqMatrix.getValue(j, i));
				}
			}
			
		    for (int i : testItems) {
		    	if (predictions.getValue(i) > 0) {
		    		result.setValue(i, predictions.getValue(i) / frequencies.getValue(i));
		    	}
		    	else {
		    		result.setValue(i, (maxValue + minValue) / 2);
		    	}
		    }
		}
		
		return result;
	}
}