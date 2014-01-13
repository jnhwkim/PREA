package prea.recommender;

import prea.data.structure.SparseMatrix;
import prea.recommender.baseline.Average;
import prea.recommender.baseline.Constant;
import prea.recommender.baseline.Random;
import prea.util.EvaluationMetrics;

/**
 * A unit test module for user-defined custom recommenders.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class UnitTest {
	private SparseMatrix usedRateMatrix;
	private SparseMatrix usedTestMatrix;
	private SparseMatrix preservedRateMatrix;
	private SparseMatrix preservedTestMatrix;
	private Recommender targetRecommender;
	
	/**
	 * Construct an instance of unit test module.
	 * 
	 * @param r The recommender to be tested.
	 * @param rm The rating matrix with train data.
	 * @param tm The rating matrix with test data.
	 */
	public UnitTest(Recommender r, SparseMatrix rm, SparseMatrix tm) {
		targetRecommender = r;
		usedRateMatrix = rm;
		usedTestMatrix = tm;
		preservedRateMatrix = new SparseMatrix(rm);
		preservedTestMatrix = new SparseMatrix(tm);
	}
	
	/**
	 * Verify whether the recommender implemented correctly.
	 *  1) Check whether the recommender illegally alters train data.
	 *  2) Check whether the recommender illegally alters test data.
	 *  3) Compare to several baselines, which should beat in general.
	 */
	public void check() {
		// Find out properties of input data:
		int[] len = preservedRateMatrix.length();
		int userCount = len[0] - 1;
		int itemCount = len[1] - 1;
		double maxValue = preservedRateMatrix.max();
		double minValue = preservedRateMatrix.min();
		
		// Learn with the target model:
		targetRecommender.buildModel(usedRateMatrix);
		EvaluationMetrics targetResult = targetRecommender.evaluate(usedTestMatrix);
		double targetMAE = targetResult.getMAE();
		double targetRMSE = targetResult.getRMSE();
		
		boolean error = false;
		
		// Test 1: Check usedRateMatrix == preservedRateMatrix:
		for (int u = 1; u <= userCount; u++) {
			int[] preservedItemList = preservedRateMatrix.getRowRef(u).indexList();
			int[] usedItemList = usedRateMatrix.getRowRef(u).indexList();
			
			if (preservedItemList.length > usedItemList.length) {
				System.out.println("Error: Existing train data was removed! (u = " + u + ")");
			}
			else if (preservedItemList.length < usedItemList.length) {
				System.out.println("Error: New train data was inserted! (u = " + u + ")");
			}
			
			if (preservedItemList != null) {
				for (int i : preservedItemList) {
					double preservedValue = preservedRateMatrix.getValue(u, i);
					double usedValue = usedRateMatrix.getValue(u, i);
					
					if (preservedValue != usedValue) {
						System.out.println("Error: Existing train data was altered! (u = " + u + ", i = " + i + ")");
						error = true;
					}
				}
			}
		}
		
		// Test 2: Check usedTestMatrix == preservedTestMatrix:
		for (int u = 1; u <= userCount; u++) {
			int[] preservedItemList = preservedTestMatrix.getRowRef(u).indexList();
			int[] usedItemList = usedTestMatrix.getRowRef(u).indexList();
			
			if (preservedItemList != null) {
				if (preservedItemList.length > usedItemList.length) {
					System.out.println("Error: Existing test data was removed! (u = " + u + ")");
				}
				else if (preservedItemList.length < usedItemList.length) {
					System.out.println("Error: New test data was inserted! (u = " + u + ")");
				}
			
				for (int i : preservedItemList) {
					double preservedValue = preservedTestMatrix.getValue(u, i);
					double usedValue = usedTestMatrix.getValue(u, i);
					
					if (preservedValue != usedValue) {
						System.out.println("Error: Existing test data was altered! (u = " + u + ", i = " + i + ")");
						error = true;
					}
				}
			}
		}
		
		// Test 3: Compare to several baselines:
		Constant constant = new Constant(userCount, itemCount, maxValue, minValue, (maxValue + minValue) / 2);
		EvaluationMetrics constantResult = constant.evaluate(preservedTestMatrix);
		double constantMAE = constantResult.getMAE();
		double constantRMSE = constantResult.getRMSE();
		
		Average average = new Average(userCount, itemCount, maxValue, minValue);
		average.buildModel(preservedRateMatrix);
		EvaluationMetrics averageResult = average.evaluate(preservedTestMatrix);
		double averageMAE = averageResult.getMAE();
		double averageRMSE = averageResult.getRMSE();
		
		Random random = new Random(userCount, itemCount, maxValue, minValue);
		EvaluationMetrics randomResult = random.evaluate(preservedTestMatrix);
		double randomMAE = randomResult.getMAE();
		double randomRMSE = randomResult.getRMSE();
		
		if (targetMAE > constantMAE) {
			System.out.println("Warning: Your recommender works poorer than Constant baseline in MAE. (Yours: " + targetMAE + ", Constant: " + constantMAE + ")");
			error = true;
		}
		if (targetRMSE > constantRMSE) {
			System.out.println("Warning: Your recommender works poorer than Constant baseline in RMSE. (Yours: " + targetRMSE + ", Constant: " + constantRMSE + ")");
			error = true;
		}
		if (targetMAE > averageMAE) {
			System.out.println("Warning: Your recommender works poorer than Average baseline in MAE. (Yours: " + targetMAE + ", Average: " + averageMAE + ")");
			error = true;
		}
		if (targetRMSE > averageRMSE) {
			System.out.println("Warning: Your recommender works poorer than Average baseline in RMSE. (Yours: " + targetRMSE + ", Average: " + averageRMSE + ")");
			error = true;
		}
		if (targetMAE > randomMAE) {
			System.out.println("Warning: Your recommender works poorer than Random baseline in MAE. (Yours: " + targetMAE + ", Random: " + randomMAE + ")");
			error = true;
		}
		if (targetRMSE > randomRMSE) {
			System.out.println("Warning: Your recommender works poorer than Random baseline in RMSE. (Yours: " + targetRMSE + ", Random: " + randomRMSE + ")");
			error = true;
		}
		
		if (!error) {
			System.out.println("Pass: No error was occurred with your recommender.");
		}
	}
}