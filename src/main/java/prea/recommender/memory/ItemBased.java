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
 * The class implementing item-based neighborhood method,
 * predicting by referring to rating matrix for each query.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class ItemBased extends MemoryBasedRecommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Average of ratings for each item. */
	public SparseVector itemRateAverage;
	/** Indicating whether the pre-calculated item similarity file is used. */
	public boolean itemSimilarityPrefetch;
	/** The name of pre-calculated item similarity file, if it is used. */
	public String itemSimilarityFileName;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct an item-based model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param ns The neighborhood size.
	 * @param sim The method code of similarity measure.
	 * @param df Indicator whether to use default values.
	 * @param dv Default value if used.
	 * @param ira The average of ratings for each item.
	 * @param isp Whether the pre-calculated item similarity file is used.
	 * @param isfn The name of pre-calculated item similarity file, if it is used.
	 */
	public ItemBased(int uc, int ic, int max, int min, int ns, int sim, boolean df, double dv, SparseVector ira, boolean isp, String isfn) {
		super(uc, ic, max, min, ns, sim, df, dv);
		
		itemRateAverage = ira;
		itemSimilarityPrefetch = isp;
		itemSimilarityFileName = isfn;
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
		
		for (int u = 1; u <= userCount; u++) {
			int[] testItems = testMatrix.getRowRef(u).indexList();
			
			if (testItems != null) {
				SparseVector predictedForUser;
				
				if (itemSimilarityPrefetch) {
					SparseMatrix itemSim = readItemSimData(testItems);
					predictedForUser = predict(u, testItems, neighborSize, itemSim);
				}
				else {
					predictedForUser = predict(u, testItems, neighborSize, null);
				}
				
				if (predictedForUser != null) {
					for (int i : predictedForUser.indexList()) {
						predicted.setValue(u, i, predictedForUser.getValue(i));
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
	 * @param itemSim The similarity vector between the target user and all the other users.
	 * @return The predicted ratings for each item.
	 */
	private SparseVector predict(int userNo, int[] testItemIndex, int k, SparseMatrix itemSim) {
		if (testItemIndex == null)
			return null;
		
		SparseVector c = new SparseVector(itemCount+1);

		for (int i : testItemIndex) {
			SparseVector a = rateMatrix.getColRef(i);
			
			// calculate similarity of every item to item i:
			double[] sim = new double[itemCount];
			int[] index = new int[itemCount];
			int[] similarItems = new int[k];
			int tmpIdx = 0;
			
			for (int j = 1; j <= itemCount; j++) {
				if (rateMatrix.getValue(userNo, j) > 0.0) {
					double similarityMeasure;

					if (itemSimilarityPrefetch) {
						if (i < j)
							similarityMeasure = itemSim.getValue(i, j);
						else
							similarityMeasure = itemSim.getValue(j, i);
					}
					else {
						SparseVector b = rateMatrix.getColRef(j);
						similarityMeasure = similarity (false, a, b, itemRateAverage.getValue(i), itemRateAverage.getValue(j), similarityMethod);
					}
					
					if (similarityMeasure > 0.0) {
						sim[tmpIdx] = similarityMeasure;
						index[tmpIdx] = j;
						tmpIdx++;
					}
				}
			}
			
			// find k most similar items:
			Sort.kLargest(sim, index, 0, tmpIdx-1, neighborSize);
			
			int similarItemCount = 0;
			for (int j = 0; j < k; j++) {
				if (sim[j] > 0) { // sim[j] is already sorted!
					similarItems[j] = index[j];
					similarItemCount++;
				}
			}
			
			if (similarItemCount > 0) {
				// estimate preference of item i by user a:
				double estimated = estimation(i, userNo, similarItems, similarItemCount, sim, WEIGHTED_SUM);
				
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
	 * @param method The code of estimation method. It should be either WEIGHTED_SUM or SIMPLE_WEIGHTED_AVG.
	 * 
	 * @return The estimated rating value.
	 */
	private double estimation(int activeIndex, int targetIndex, int[] ref, int refCount, double[] refWeight, int method) {
		double sum = 0.0;
		double weightSum = 0.0;
		double result = 0.0;
		
		if (method == WEIGHTED_SUM) { // Weighted Sum of Others' rating
			double activeAvg = itemRateAverage.getValue(activeIndex);
			
			for (int u = 0; u < refCount; u++) {
				double refAvg = itemRateAverage.getValue(ref[u]);
				double ratedValue = rateMatrix.getValue(targetIndex, ref[u]);
				
				if (ratedValue > 0.0) {
					sum += ((ratedValue - refAvg) * refWeight[u]);
					weightSum += refWeight[u];
				}
			}
			
			result = activeAvg + sum / weightSum;
		}
		else if (method == SIMPLE_WEIGHTED_AVG) { // Simple Weighted Average
			for (int u = 0; u < refCount; u++) {
				double ratedValue = rateMatrix.getValue(targetIndex, ref[u]);
				
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
	
	/*========================================
	 * File I/O
	 *========================================*/
	/**
	 * Read the pre-calculated item similarity data file.
	 * Make sure that the similarity file is compatible with the split file you are using,
	 * for a fair comparison.
	 * 
	 * @param validationItemSet The list of items which will be used for validation.
	 * @return The item similarity matrix.
	 */
	private SparseMatrix readItemSimData(int[] validationItemSet) {
		SparseMatrix itemSimilarity = new SparseMatrix (itemCount+1, itemCount+1);
		
		try {
			FileInputStream stream = new FileInputStream(itemSimilarityFileName);
			InputStreamReader reader = new InputStreamReader(stream);
			BufferedReader buffer = new BufferedReader(reader);
			
			// validationItemSet needs to be sorted at here!!
			Sort.quickSort(validationItemSet, 0, validationItemSet.length - 1, true);

			String line;
			int lineNo = 1;
			int validIdx = 0; // validation item index
			while((line = buffer.readLine()) != null && !line.equals("TT_EOF")) {
				int itemIdx = validationItemSet[validIdx];
				
				if (lineNo == itemIdx) {
					StringTokenizer st = new StringTokenizer (line);
					
					int idx = 1;
					while (st.hasMoreTokens()) {
						double sim = Double.parseDouble(st.nextToken()) / 10000;
						
						if (sim != 0.0) {
							itemSimilarity.setValue(itemIdx, idx, sim);
						}
						
						idx++;
					}
					
					validIdx++;
					
					if (validIdx >= validationItemSet.length)
						break;
				}
				
				lineNo++;
			}
			
			stream.close();
		}
		catch (IOException ioe) {
			System.out.println ("No such file.");
		}
		
		return itemSimilarity;
	}
}
