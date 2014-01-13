package prea.recommender.etc;
import prea.data.structure.DenseMatrix;
import prea.data.structure.DenseVector;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.recommender.Recommender;
import prea.util.EvaluationMetrics;

/**
 * This is a class implementing Fast Nonparametric Principal Component Analysis (NPCA).
 * Technical detail of the algorithm can be found in
 * Kai Yu et al, Fast Nonparametric Matrix Factorization for Large-scale Collaborative Filtering,
 * Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval, 2009.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class FastNPCA implements Recommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	
	/** Rating matrix for items which will be used during the validation phase.
	 * Not allowed to refer during training phase. */
	private SparseMatrix validationMatrix;
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public double minValue;
	/** Maximum number of iteration. */
	public int maxIter;
	/** Proportion of dataset, using for validation purpose. */
	public double validationRatio;
	
	/** Indicator whether to show progress of iteration. */
	public boolean showProgress = false;
	
	private SparseMatrix K;
	private SparseVector mu;
	
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
	 * @param vr The proportion of dataset which will be used for validation. 
	 * @param iter The maximum number of iteration.
	 */
	public FastNPCA(int uc, int ic, double max, double min, double vr, int iter) {
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
		validationRatio = vr;
		maxIter = iter;
		
		K = new SparseMatrix(itemCount+1, itemCount+1);
		mu = new SparseVector(itemCount+1);
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
		
		makeValidationSet(validationRatio);
		
		double totalAverage = rateMatrix.average();
		double prevErr = 99999;
		double currErr = 9999;
		
		// Simple Initialization:
		double rateSum = 0.0;
		for (int i = 1; i <= itemCount; i++) {
			SparseVector ci = rateMatrix.getColRef(i);
			int[] rateList = ci.indexList();
			if (rateList != null) {
				for (int r : rateList) {
					rateSum += Math.pow(rateMatrix.getValue(r, i) - totalAverage, 2);
				}
			}
		}

		K = SparseMatrix.makeIdentity(itemCount+1);
		K.selfScale(rateSum / (double) rateMatrix.itemCount());
		mu.initialize(totalAverage);
		
		// Iterative EM Algorithm:
		int round = 1;
		
		while (prevErr > currErr /*&& Math.abs(prevErr - currErr) > 0.0001*/ && round < maxIter) {
			int MValid = 0;
			SparseMatrix BigInv = new SparseMatrix(itemCount+1, itemCount+1);
			SparseVector BigInvF = new SparseVector(itemCount+1);
			
			// E-Step: compute the sufficient statistics of posterior
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = rateMatrix.getRow(u).indexList();
				
				if (itemList != null) {
					// If has too many rating, randomly choose:
					int itemThreshold = 300;
					int[] itemListLimited = new int[Math.min(itemThreshold, itemList.length)];
					
					if (itemList.length > itemThreshold * 2) {
						int curSize = 0;
						while (curSize < itemThreshold) {
							int idx = (int) (Math.random() * curSize);
							int temp = itemList[curSize];
							itemList[curSize] = itemList[idx];
							itemList[idx] = temp;
							curSize++;
						}
						System.arraycopy(itemList, 0, itemListLimited, 0, itemListLimited.length);
					}
					else if (itemList.length > itemThreshold) {
						int curSize = itemList.length;
						while (curSize > itemThreshold) {
							int idx = (int) (Math.random() * curSize);
							itemList[idx] = itemList[curSize-1];
							curSize--;
						}
						System.arraycopy(itemList, 0, itemListLimited, 0, itemListLimited.length);
					}
					else {
						itemListLimited = itemList;
					}
					
					MValid++;
					SparseVector itemRates = rateMatrix.getRow(u);
					SparseMatrix KII_inv = K.partInverse(itemListLimited);
					itemRates = itemRates.partMinus(mu, itemListLimited);
					SparseVector t = KII_inv.partTimes(itemRates, itemListLimited);
					BigInv = BigInv.partPlus(t.partOuterProduct(t, itemListLimited).partMinus(KII_inv, itemListLimited), itemListLimited);
					BigInvF = BigInvF.partPlus(t, itemListLimited);
				}
			}
			
			// check if the validation error increases, if yes, stop the iterations
			
			// M-step: compute K and mu
			DenseMatrix DK = K.toDenseMatrix();
			DenseMatrix DB = BigInv.toDenseMatrix();
			
			SparseVector f = K.times(BigInvF);
			
			DenseMatrix result = DK.times(DB).times(DK).scale(1/(double) userCount).plus(DK);
			K = result.toSparseMatrix();
			
			mu = mu.plus(f.scale(1/(double) MValid));
			
			
			// show progress:
			prevErr = currErr;
			
			double err = 0.0;
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = validationMatrix.getRowRef(u).indexList();
				
				if (itemList != null) {
					SparseVector estimate = getEstimation(u, itemList);
					
					for (int i : itemList) {
						double Aij = validationMatrix.getValue(u, i);
						double Bij = estimate.getValue(i);
						err += Math.pow(Aij - Bij, 2);
					}
				}
			}
			
			currErr = err / (double) validationMatrix.itemCount();
			
			if (showProgress)
				System.out.println(round + "\t" + currErr);
			
			round++;
		}
		
		restoreValidationSet();
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
		
		for (int u = 1; u <= userCount; u++) {
			int[] testItems = testMatrix.getRowRef(u).indexList();
			
			if (testItems != null) {
				SparseVector predictedForUser = getEstimation(u, testItems);
			
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
	 * Estimate of ratings for a given user and a set of test items.
	 * 
	 * @param u The user number.
	 * @param testItems The list of items to be predicted.
	 * 
	 * @return A list containing predicted rating scores.
	 */
	private SparseVector getEstimation(int u, int[] testItems) {
		SparseVector result = new SparseVector(itemCount+1);
		
		int[] ratedItems = rateMatrix.getRowRef(u).indexList();
		
		if (ratedItems == null) {
			for (int t = 0; t < testItems.length; t++) {
				result.setValue(testItems[t], 3.0);
			}
		}
		else {
			DenseMatrix KII = K.toDenseSubset(ratedItems).inverse();
			DenseVector vecI = rateMatrix.getRowRef(u).minus(mu).toDenseSubset(ratedItems);
			DenseMatrix KJI = K.toDenseSubset(testItems, ratedItems);
			DenseVector offset = KJI.times(KII.times(vecI));
			
			for (int t = 0; t < testItems.length; t++) {
				double estimate = offset.getValue(t) + mu.getValue(testItems[t]);
				
				// rating should be located between minValue and maxValue:
				if (estimate < minValue)
					estimate = minValue;
				else if (estimate > maxValue)
					estimate = maxValue;
				
				result.setValue(testItems[t], estimate);
			}
		}
		
		return result;
	}
	
	/*========================================
	 * Train/Validation set management
	 *========================================*/
	/**
	 * Items which will be used for validation purpose are moved from rateMatrix to validationMatrix.
	 * 
	 * @param validationRatio Proportion of dataset, using for validation purpose.
	 */
	private void makeValidationSet(double validationRatio) {
		validationMatrix = new SparseMatrix(userCount+1, itemCount+1);
		
		int validationCount = (int) (rateMatrix.itemCount() * validationRatio);
		while (validationCount > 0) {
			int index = (int) (Math.random() * userCount) + 1;
			SparseVector row = rateMatrix.getRowRef(index);
			int[] itemList = row.indexList();
			
			if (itemList != null && itemList.length > 5) {
				int index2 = (int) (Math.random() * itemList.length);
				validationMatrix.setValue(index, index2, rateMatrix.getValue(index, itemList[index2]));
				rateMatrix.setValue(index, itemList[index2], 0.0);
				
				validationCount--;
			}
		}
	}
	
	/** Items in validationMatrix are moved to original rateMatrix. */
	private void restoreValidationSet() {
		for (int i = 1; i <= userCount; i++) {
			SparseVector row = validationMatrix.getRowRef(i);
			int[] itemList = row.indexList();
			
			if (itemList != null) {
				for (int j : itemList) {
					rateMatrix.setValue(i, j, validationMatrix.getValue(i, j));
					//validationMatrix.setValue(i, j, 0.0);
				}
			}
		}
	}
}