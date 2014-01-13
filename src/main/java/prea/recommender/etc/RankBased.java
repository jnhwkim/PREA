package prea.recommender.etc;

import java.util.Arrays;

import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;
import prea.recommender.Recommender;
import prea.util.EvaluationMetrics;
import prea.util.Sort;
import prea.util.Loss;
import prea.util.Distance;

/**
 * This is a class implementing rank-based collaborative filtering.
 * Technical detail of the algorithm can be found in
 * M. Sun and G. Lebanon and P. Kidwell, Estimating Probabilities in Recommendation Systems,
 * Proceedings of the fourteenth international conference on Artificial Intelligence and Statistics, 2011.
 * 
 * @author Mingxuan Sun
 * @since 2012. 4. 20
 * @version 1.1
 */
public class RankBased implements Recommender {
	/*========================================
	 * Method Names
	 *========================================*/
	// Loss functions
	/** Mean loss function */
	public static final int MEAN_LOSS = 2001;
	/** Asymmetric loss function */
	public static final int ASYMM_LOSS = 2002;

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
	public int maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public int minValue;
	/** Kernel bandwidth. */
	public double kernelWidth;
	/** Parsed prb for each user, 3d array with l rating levels by m testing users by the number of rated items. */
	public double[][][] prbMatricesVal;
	/** Parsed index for each user, 2d array with m testing users by the number of rated items. */
	public int[][] prbMatricesIndTrain;
	/** The number of tie structure for each user with size m testing users by l raing levels. */
	public int[][] numTieCount;
	/** The probability array of each rating for testing items and users. */
	public SparseMatrix[] predictedArray;
	/** The type of loss function. */
	private int lossType;


	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a rank-based model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param kw The kernel width for density estimation
	 * @param lt Loss function used for prediction. It should be either MEAN_LOSS or ASYMM_LOSS.
	 */
	public RankBased(int uc, int ic, double max, double min, double kw, int lt) {
		userCount = uc;
		itemCount = ic;
		
		// This algorithm assumes that ratings are integers.
		maxValue = (int) max;
		minValue = (int) min;
		kernelWidth = kw;
		
		if (lt == MEAN_LOSS || lt == ASYMM_LOSS) {
			lossType = lt;
		}
		else {
			lossType = MEAN_LOSS;
		}	
	}


	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Predict the probabilities using the rank-based algorithm with the given test data.
	 * The probability of each test item for each user will be filled in predictedArray.
	 * 
	 * @param rm The rating matrix with train data.
	 */
	@Override
	public void buildModel(SparseMatrix rm) {
		rateMatrix = rm;
		
		// Initialization:
		prbMatricesVal = new double[maxValue - minValue + 2][userCount][];
		prbMatricesIndTrain = new int[userCount][];

		for (int i = 0; i < userCount; i++) {
			int k = (rm.getRowRef(i+1)).itemCount();
			prbMatricesIndTrain[i] = new int[k];
			for (int l = 0; l < maxValue - minValue + 2; l++) {
				prbMatricesVal[l][i] = new double[k+1];
			}
		}

		constructProbability();		
		numTieCount = new int[userCount][maxValue - minValue + 1];
		constructTie();		

		
	}
	
	/** 
	 * Construct the probability (average rank) structure for each user's ranking.
	 */
	private void constructProbability() {
		for (int u = 1; u <= userCount; u++) {
			SparseVector trainRateList = rateMatrix.getRowRef(u);
			
			// The number of train items is less than 1, averageRank of unknown testid is always 0.5
			if (trainRateList.itemCount() < 1) {
			    for (int l = minValue; l <= maxValue; l++){
				prbMatricesVal[l-minValue+1][u-1][0] = 0.5;
			    }
			    continue;
			}			
			int[] itemID = trainRateList.indexList();
			double[] score = trainRateList.valueList();
			//score should be larger than zero, score has the same length as itemID
			int k = itemID.length;
			double[] uPrb = new double[score.length];
			Distance.computeAverageRank(score, uPrb);
			//sort id from smallest to biggest here
			int[] itemIDclone = itemID.clone();
			int[] id = new int[k];
			for (int i = 0 ; i < id.length; i++){
				id[i]=i;
			}
			Sort.quickSort(itemIDclone, id, 0, id.length-1, true);
			for (int i = 0; i < k; i++) {
				prbMatricesIndTrain[u-1][i] = itemID[id[i]];
				prbMatricesVal[0][u-1][i] = uPrb[id[i]];
			}    
			double[] scoreE = new double[k+1];
			System.arraycopy(score, 0, scoreE, 0, k);
			for (int l = minValue; l <= maxValue; l++){
				scoreE[k] = (double)l;
				double[] uPrbE = new double[scoreE.length];
				Distance.computeAverageRank(scoreE, uPrbE);
				for (int i = 0; i < k; i++) {
					prbMatricesVal[l-minValue+1][u-1][i] = uPrbE[id[i]];
				}
				prbMatricesVal[l-minValue+1][u-1][k] = uPrbE[k];
			}
		}
	}
	
	/**
	 * Construct the tie structure for each user's ranking.
	 */
	private void constructTie() {
		for (int u = 1; u <= userCount; u++) {
			SparseVector trainRateList = rateMatrix.getRowRef(u);
			if(trainRateList.itemCount() > 0){
			    double[] score = trainRateList.valueList();
			    for(int i = 0; i < score.length; i++){
				int l = (int)score[i];
				numTieCount[u-1][l-minValue] += 1;
			    }
			}
		}
	}

	/**
	 * Compute the distance between the testing user and all the other training users with training items.
	 * 
	 * @param userId The user ID.
	 * @param dist The distance which will be computed.
	 * @param k The neighborhood size.
	 * @param indexK The index of nearest neighbor which will be computed.
	 */
	private void distanceOneToAllTrain(int userId, double[] dist, int k, int[] indexK) {
		int[] vItemID = prbMatricesIndTrain[userId-1];
		double[] vPrb = prbMatricesVal[0][userId-1];
		int t = 0;
		for (int u = 1; u <= userCount; u++) {
			if(u != userId){
				int[] uItemID = prbMatricesIndTrain[u-1];
				double[] uPrb = prbMatricesVal[0][u-1];	
				dist[t] = Distance.distanceSpearmanParsed(uItemID, uPrb, vItemID, vPrb, itemCount);	
				indexK[t] = u;
				t++;
			}
		}

		// Find k nearest neighbors:
		Sort.kSmallest(dist, indexK, 0, userCount-2, k); 
	}
	
	/**
	 * Compute the distance between the testing user with the testing item of every possible ratings and all the other training users with training items.
	 * 
	 * @param userId The user ID.
	 * @param testId The testing item ID.
	 * @param dist The distance which will be computed.
	 * @param k The neighborhood size.
	 * @param indexK The index of nearest neighbor.
	 */
	private void distanceOneToAllTest(int userId, int testId, SparseVector[] dist, int k, int[] indexK){
		int[] vItemID = prbMatricesIndTrain[userId - 1];		
		for (int l = 1; l < prbMatricesVal.length; l++){
			// vItemID sorted increasingly
			double[] vPrb = prbMatricesVal[l][userId - 1];
			int[] vItemID_l = new int[vItemID.length + 1];
			double[] vPrb_l = new double[vItemID.length + 1];
			int index = Arrays.binarySearch(vItemID, testId);
			
			if (index < 0) {
				// Compute the insert index. Insert the new item into sortedArray.Create an array of size+1. 
				int insertIndex = -index - 1;		  		
				System.arraycopy(vItemID, 0, vItemID_l, 0, insertIndex);
				System.arraycopy(vItemID, insertIndex, vItemID_l, insertIndex+1, vItemID.length-insertIndex);
				vItemID_l[insertIndex] = testId;
				System.arraycopy(vPrb, 0, vPrb_l, 0, insertIndex);
				System.arraycopy(vPrb, insertIndex, vPrb_l, insertIndex+1, vPrb.length-1-insertIndex);
				vPrb_l[insertIndex]=vPrb[vPrb.length-1];
			
				for (int id_u = 0; id_u < k; id_u++) {
					int u = indexK[id_u];
					int[] uItemID = prbMatricesIndTrain[u-1];
					double[] uPrb = prbMatricesVal[0][u-1];	
					double value = Distance.distanceSpearmanParsed(vItemID_l, vPrb_l, uItemID, uPrb, itemCount);
					dist[l-1].setValue(u,value);
				}
			}	       	
		}
	}

	/**
	 * Predict ratings for a given user and a given test item, by rank-based CF algorithm.
	 * 
	 * @param userId The user ID.
	 * @param testItemId The id of item whose ratings will be predicted.
	 * @param ker The kernelwidth.
	 * @param k The number of nearest training users.
	 * @param index The index of near k users.
	 * @param distL0 The distance between the test user and all the other k training users.
	 * @param distL  The distance between the test user with testing item of each possible ratings and all the other k training users.
	 * @param predictedArray The array will be filled in with the probabilities of each rating for the user u and the item.
	 */
	private void rankBasedPerUser(int userId, int testItemId, double ker, int k, int[] index, double[] distL0, SparseVector[] distL, SparseMatrix[] predictedArray) {
		double varU = 0.3;
		double varR = 0.0000001 * ker;
		int lnum = maxValue-minValue+1;
		double[] prbl = new double[lnum];
		
		// find k most similar users for this user:
		double dmax_u = distL0[k-1];
		double dmin_u = distL0[0];
		double dvar = dmax_u - dmin_u;
		
		for (int i = 0; i < k; i++){
			double d = (distL0[i]-dmin_u)/(dvar+0.000001);
			double prbraw = Math.exp(-(d*d)/varU);
			if (rateMatrix.getValue(index[i],testItemId)==0){
			    prbraw = 0;
			}
			double dmin = 1.0;
			for (int l = 1; l <= lnum; l++){
				double temp = distL[l-1].getValue(index[i]);
				if( temp < dmin){
					dmin = temp;
				}
			}
			double s = 0;
			for (int l = 1; l <= lnum; l++){	
				d = distL[l-1].getValue(index[i]) - dmin;
				s = s + Math.exp(-(d*d)/varR);

			}
			for (int l = 1; l <= lnum; l++){	
				d = distL[l-1].getValue(index[i]) - dmin;
				prbl[l-1] += prbraw * Math.exp(-(d*d)/varR)/s;
			
			}				    
		}
		
		double sum = 0;
		for (int l = 0; l < lnum; l++){
		    prbl[l] = prbl[l] * (double)(numTieCount[userId-1][l]+1);
			sum += prbl[l];
		}
		for (int l = 0; l < lnum; l++){
		    if (sum == 0) {
		    	predictedArray[l].setValue(userId, testItemId, 1.0/lnum);
		    }
		    else {
		    	predictedArray[l].setValue(userId, testItemId, prbl[l]/sum);
		    }
		}
	}

	/*========================================
	 * Prediction
	 *========================================*/
	/**
	 * Evaluate the rank-based CF algorithm with the given probabilites and loss function.
	 * 
	 * @param testMatrix The rating matrix with test data.
	 * 
	 * @return The result of evaluation, such as MAE, RMSE, and rank-score.
	 */
	public EvaluationMetrics evaluate(SparseMatrix testMatrix) {
		SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
		getEstimation(testMatrix);

		for (int u = 1; u <= userCount; u++) {
			SparseVector realRateList = testMatrix.getRowRef(u);

			if (realRateList.itemCount() > 0){
				int[] testItemList = realRateList.indexList();

				for (int i = 0; i < testItemList.length; i++){
					int itemId = testItemList[i];
					double estimate = predict(u, itemId);
					predicted.setValue(u, itemId, estimate);
				}
			}
		}

		return new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);	  
	}
	
	/**
	 * Estimate ratings for (user, item) pairs in test data matrix.
	 * 
	 * @param testMatrix The rating matrix with test data.
	 */
	private void getEstimation(SparseMatrix testMatrix) {
		predictedArray = new SparseMatrix[maxValue - minValue + 1];
		for (int i = 0; i < maxValue - minValue + 1; i++) {
			predictedArray[i] = new SparseMatrix((testMatrix.length())[0],(testMatrix.length())[1]);
		}
		
		// Learning:
		for (int u = 1; u <= userCount; u++) {
			int[] testItems = testMatrix.getRowRef(u).indexList();
			
			if (testItems != null) {
				// Calculate the distance of every possible rate with every user:
				// Find the k nearest neighbors:
				int nearK = Math.min(500, userCount-1);	
				int[] indexK = new int[userCount+1];
				double[] distL0 = new double [userCount+1];			
				SparseVector[] distL = new SparseVector[maxValue-minValue+1];
				
				for (int i = 0; i < maxValue - minValue + 1; i++){
					distL[i] = new SparseVector(userCount+1);
				}
				
				distanceOneToAllTrain(u, distL0, nearK, indexK);
				
				for (int myi = 0; myi < testItems.length; myi++) {				   
					distanceOneToAllTest(u, testItems[myi], distL, nearK, indexK);		   			   
					rankBasedPerUser(u, testItems[myi], kernelWidth, nearK, indexK, distL0, distL, predictedArray);
				}
			}
		}
	}
	
	/**
	 * Compute the rating for a given user and a test item,
	 * which minimizes the given loss.
	 * 
	 * @param userId The target user.
	 * @param itemId The target test item.
	 * 
	 * @return The predicted rating.
	 */
	private double predict(int userId, int itemId) {
		double result = 0.0;
		
		if (this.lossType == MEAN_LOSS) {
			double mean = 0;
			
			for(int l = minValue; l <= maxValue; l++){
				mean += (double) l * predictedArray[l - minValue].getValue(userId, itemId);
			}
			
			result = mean;
		}
		else if (this.lossType == ASYMM_LOSS) {
			double[] loss = new double[maxValue - minValue + 1];
			
			for(int l = minValue; l <= maxValue; l++){
				double s = 0;
				double pred = (double) l;
				
				for(int j = minValue; j <= maxValue; j++){
					double prb = predictedArray[j - minValue].getValue(userId, itemId);
					double real = (double) j;
					double c = Loss.asymmetricLoss(real, pred, minValue, maxValue);
					s += c * prb;
				}
				
				loss[l - minValue] = s;
			}
			
			int id = 0;
			double min_val = loss[id];
			for (int i= 0; i < loss.length; i++){
				if(loss[i] < min_val){
					id = i;
					min_val = loss[i];
				}
			}

			result = (double) minValue + id;
		}
		
		return result;
	}
}