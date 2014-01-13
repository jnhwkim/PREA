package prea.recommender.memory;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.StringTokenizer;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.util.EvaluationMetrics;
import prea.util.Sort;

/**
 * The class implementing user-based neighborhood method,
 * predicting by referring to rating matrix for each query.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class UserBased extends MemoryBasedRecommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Average of ratings for each user. */
	public SparseVector userRateAverage;
	/** Indicating whether the pre-calculated user similarity file is used. */
	public boolean userSimilarityPrefetch;
	/** The name of pre-calculated user similarity file, if it is used. */
	public String userSimilarityFileName;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a user-based model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param ns The neighborhood size.
	 * @param sim The method code of similarity measure.
	 * @param df Indicator whether to use default values.
	 * @param dv Default value if used.
	 * @param ura The average of ratings for each user. 
	 * @param usp Whether the pre-calculated user similarity file is used.
	 * @param usfn The name of pre-calculated user similarity file, if it is used.
	 */
	public UserBased(int uc, int ic, int max, int min, int ns, int sim, boolean df, double dv, SparseVector ura, boolean usp, String usfn) {
		super(uc, ic, max, min, ns, sim, df, dv);
		
		userRateAverage = ura;
		userSimilarityPrefetch = usp;
		userSimilarityFileName = usfn;
	}
	
	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the designated algorithm with the given test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	@Override
	public EvaluationMetrics evaluate(SparseMatrix testMatrix) {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		
		if (userSimilarityPrefetch) {
			try {
				FileInputStream stream = new FileInputStream(userSimilarityFileName);
				InputStreamReader reader = new InputStreamReader(stream);
				BufferedReader buffer = new BufferedReader(reader);
				
				String line;
				for (int u = 1; u <= userCount; u++) {
					line = buffer.readLine();
					int[] testItems = testMatrix.getRowRef(u).indexList();
					
					if (testItems != null) {
						// Parse similarity
						double[] userSim = new double[userCount+1];
						StringTokenizer st = new StringTokenizer (line);
						int idx = 1;
						while (st.hasMoreTokens()) {
							double sim = Double.parseDouble(st.nextToken()) / 10000;
							
							if (sim != 0.0) {
								userSim[idx] = sim;
							}
							
							idx++;
						}
						
						// Prediction
						SparseVector predictedForUser = predict(u, testItems, neighborSize, userSim);
						
						if (predictedForUser != null) {
							for (int i : predictedForUser.indexList()) {
								predicted.setValue(u, i, predictedForUser.getValue(i));
							}
						}
					}
				}
				
				stream.close();
			}
			catch (IOException ioe) {
				System.out.println ("No such file.");
			}
		}
		else {
			for (int u = 1; u <= userCount; u++) {
				int[] testItems = testMatrix.getRowRef(u).indexList();
				
				if (testItems != null) {
					SparseVector predictedForUser = predict(u, testItems, neighborSize, null);
					
					if (predictedForUser != null) {
						for (int i : predictedForUser.indexList()) {
							predicted.setValue(u, i, predictedForUser.getValue(i));
						}
					}
				}
			}
		}
		
		return new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
	}
	
	/**
	 * Predict ratings for a given user regarding given set of items, by user-based CF algorithm.
	 * 
	 * @param userNo The user ID.
	 * @param testItemIndex The list of items whose ratings will be predicted.
	 * @param k The neighborhood size.
	 * @param userSim The similarity vector between the target user and all the other users.
	 * @return The predicted ratings for each item.
	 */
	private SparseVector predict(int userNo, int[] testItemIndex, int k, double[] userSim) {
		if (testItemIndex == null)
			return null;
		
		double[][] sim = new double[testItemIndex.length][userCount];
		int[][] index = new int[testItemIndex.length][userCount];
		SparseVector a = rateMatrix.getRow(userNo);
		SparseVector c = new SparseVector(itemCount+1);
		double a_avg = a.average();
		
		// calculate similarity with every user:
		int[] tmpIdx = new int[testItemIndex.length];
		for (int u = 1; u <= userCount; u++) {
			SparseVector b = rateMatrix.getRowRef(u);
			double similarityMeasure;
			
			if (userSimilarityPrefetch) {
				similarityMeasure = userSim[u];
			}
			else {
				similarityMeasure = similarity(true, a, b, a_avg, userRateAverage.getValue(u), similarityMethod);
			}
			
			if (similarityMeasure > 0.0) {
				for (int t = 0; t < testItemIndex.length; t++) {
					if (b.getValue(testItemIndex[t]) > 0.0) {
						sim[t][tmpIdx[t]] = similarityMeasure;
						index[t][tmpIdx[t]] = u;
						tmpIdx[t]++;
					}
				}
			}
		}
		
		// Estimate rating for items in test set:
		for (int t = 0; t < testItemIndex.length; t++) {
			// find k most similar users:
			Sort.kLargest(sim[t], index[t], 0, tmpIdx[t]-1, neighborSize);
			
			int[] similarUsers = new int[k];
			int similarUserCount = 0;
			for (int i = 0; i < k; i++) {
				if (sim[t][i] > 0) { // sim[t][i] is already sorted!
					similarUsers[i] = index[t][i];
					similarUserCount++;
				}
			}
			
			int i = testItemIndex[t];
			if (similarUserCount > 0) {
				double estimated = estimation(userNo, i, similarUsers, similarUserCount, sim[t], WEIGHTED_SUM);
				
				// NaN check: it happens that no similar user has rated on item i, then the estimate is NaN.
				if (!Double.isNaN(estimated)) {
					c.setValue(i, estimated);
				}
				else {
					c.setValue(i, (maxValue + minValue) / 2);
				}
			}
			else {
				c.setValue(i, (maxValue + minValue) / 2);
			}
		}
		
		return c;
	}
	
	/**
	 * Estimate a rating based on neighborhood data.
	 * 
	 * @param activeIndex The active user index for user-based CF; The item index for item-based CF.
	 * @param targetIndex The target item index for user-based CF; The user index for item-based CF.
	 * @param ref The indices of neighborhood, which will be used for estimation.
	 * @param refCount The number of neighborhood, which will be used for estimation.
	 * @param refWeight The weight of each neighborhood.
	 * @param method The code of estimation method. It can be one of the following: WEIGHTED_SUM or SIMPLE_WEIGHTED_AVG.
	 * 
	 * @return The estimated rating value.
	 */
	private double estimation(int activeIndex, int targetIndex, int[] ref, int refCount, double[] refWeight, int method) {
		double sum = 0.0;
		double weightSum = 0.0;
		double result = 0.0;
		
		if (method == WEIGHTED_SUM) { // Weighted Sum of Others' rating
			double activeAvg = userRateAverage.getValue(activeIndex);
			
			for (int u = 0; u < refCount; u++) {
				double refAvg = userRateAverage.getValue(ref[u]);
				double ratedValue = rateMatrix.getValue(ref[u], targetIndex);
				
				if (ratedValue > 0.0) {
					sum += ((ratedValue - refAvg) * refWeight[u]);
					weightSum += refWeight[u];
				}
			}
			
			result = activeAvg + sum / weightSum;
		}
		else if (method == SIMPLE_WEIGHTED_AVG) { // Simple Weighted Average
			for (int u = 0; u < refCount; u++) {
				double ratedValue = rateMatrix.getValue(ref[u], targetIndex);
				
				if (ratedValue > 0.0) {
					sum += (ratedValue * refWeight[u]);
					weightSum += refWeight[u];
				}
			}
			
			result = sum / weightSum;
		}
		
		// rating should be located between minValue and maxValue:
		if (result < minValue)
			result = minValue;
		else if (result > maxValue)
			result = maxValue;
		
		return result;
	}
}
