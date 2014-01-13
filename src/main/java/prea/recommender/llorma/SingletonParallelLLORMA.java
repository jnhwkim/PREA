package prea.recommender.llorma;

import prea.util.EvaluationMetrics;
import prea.util.KernelSmoothing;
import prea.recommender.Recommender;
import prea.recommender.matrix.RegularizedSVD;
import prea.data.structure.SparseVector;
import prea.data.structure.SparseMatrix;

/**
 * A class implementing Local Low-Rank Matrix Approximation.
 * Technical detail of the algorithm can be found in
 * Joonseok Lee and Seungyeon Kim and Guy Lebanon and Yoram Singer, Local Low-Rank Matrix Approximation,
 * Proceedings of the 30th International Conference on Machine Learning, 2013.
 * 
 * @author Joonseok Lee
 * @since 2013. 6. 11
 * @version 1.2
 */
public class SingletonParallelLLORMA implements Recommender {
	/*========================================
	 * Common Variables
	 *========================================*/
	/** Rating matrix for each user (row) and item (column) */
	public SparseMatrix rateMatrix;
	/** Rating matrix for test items. Not allowed to refer during training and validation phase. */
	public static SparseMatrix testMatrix;
	
	/** The number of users. */
	public int userCount;
	/** The number of items. */
	public int itemCount;
	/** Maximum value of rating, existing in the dataset. */
	public double maxValue;
	/** Minimum value of rating, existing in the dataset. */
	public double minValue;
	/** Average of ratings for each user. */
	public static SparseVector userRateAverage;
	/** Average of ratings for each item. */
	public static SparseVector itemRateAverage;
	/** Similarity matrix between users. */
	private static SparseMatrix userSimilarity;
	/** Similarity matrix between items. */
	private static SparseMatrix itemSimilarity;
	/** A global SVD model used for calculating user/item similarity. */
	public static RegularizedSVD baseline;
	
	// Local model parameters:
	/** The number of features. */
	public int featureCount;
	/** Learning rate parameter. */
	public double learningRate;
	/** Regularization factor parameter. */
	public double regularizer;
	/** Maximum number of iteration. */
	public int maxIter;
	
	// Global combination parameters:
	/** Maximum number of local models. */
	public int modelMax;
	/** Type of kernel used in kernel smoothing. */
	public int kernelType;
	/** Width of kernel used in kernel smoothing. */
	public double kernelWidth;
	/** The maximum number of threads which will run simultaneously. 
	 * We recommend not to exceed physical limit of your machine. */
	private int multiThreadCount;
	
	/** Indicator whether to show progress of iteration. */
	public boolean showProgress;
	
	/** A data structure to store intermediate evaluation result. */
	private EvaluationMetrics preservedEvalMet = null;
	
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a matrix-factorization-based model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param fc The number of features used for describing user and item profiles.
	 * @param lr Learning rate for gradient-based or iterative optimization.
	 * @param r Controlling factor for the degree of regularization. 
	 * @param iter The maximum number of iterations.
	 * @param mm The maximum number of local models.
	 * @param kt Type of kernel used in kernel smoothing.
	 * @param kw Width of kernel used in kernel smoothing.
	 * @param base A global SVD model used for calculating user/item similarity.
	 * @param tm A reference to test matrix.
	 * @param ml The number of threads which will run simultaneously.
	 * @param verbose Indicating whether to show iteration steps and train error.
	 */
	public SingletonParallelLLORMA(int uc, int ic, double max, double min, int fc, double lr, double r, int iter,
			int mm, int kt, double kw, RegularizedSVD base, SparseMatrix tm, int ml, boolean verbose) {
		userCount = uc;
		itemCount = ic;
		maxValue = max;
		minValue = min;
		
		featureCount = fc;
		learningRate = lr;
		regularizer = r;
		maxIter = iter;
		
		modelMax = mm;
		kernelType = kt;
		kernelWidth = kw;
		
		baseline = base;
		testMatrix = tm;
		multiThreadCount = ml;
		showProgress = verbose;
	}
	
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Build a model with given training set.
	 * 
	 * @param rateMatrix The rating matrix with train data.
	 */
	@Override
	public void buildModel(SparseMatrix rateMatrix) {
		// Pre-calculating similarity:
		userSimilarity = new SparseMatrix(userCount+1, userCount+1);
		itemSimilarity = new SparseMatrix(itemCount+1, itemCount+1);
		
		int completeModelCount = 0;
		
		WeakLearner[] learners = new WeakLearner[multiThreadCount];
		int[] anchorUser = new int[modelMax];
		int[] anchorItem = new int[modelMax];
		
		int modelCount = 0;
		double anchorErrCum = 0.0;
		int[] runningThreadList = new int[multiThreadCount];
		int runningThreadCount = 0;
		int waitingThreadPointer = 0;
		int nextRunningSlot = 0;
		
		double anchorErr = 0.0;
		double lowestRMSE = Double.MAX_VALUE;
		preservedEvalMet = null;
		
		SparseMatrix cumPrediction = new SparseMatrix(userCount+1, itemCount+1);
		SparseMatrix cumWeight = new SparseMatrix(userCount+1, itemCount+1);
		
		// Parallel training:
		while (modelCount < modelMax) {
			int u_t = (int) Math.floor(Math.random() * userCount) + 1;
			int[] itemList = rateMatrix.getRow(u_t).indexList();
			
			if (itemList != null) {
				if (runningThreadCount < multiThreadCount) {
					// Selecting a new anchor point:
					int idx = (int) Math.floor(Math.random() * itemList.length);
					int i_t = itemList[idx];
					
					anchorUser[modelCount] = u_t;
					anchorItem[modelCount] = i_t;
					
					// Preparing weight vectors:
					SparseVector w = kernelSmoothing(userCount+1, u_t, kernelType, kernelWidth, false);
					SparseVector v = kernelSmoothing(itemCount+1, i_t, kernelType, kernelWidth, true);

					// Starting a new local model learning:
					learners[nextRunningSlot] = new WeakLearner(modelCount, featureCount, userCount, itemCount, u_t, i_t, learningRate, regularizer, maxIter, w, v, rateMatrix);
					learners[nextRunningSlot].start();
					
					runningThreadList[runningThreadCount] = modelCount;
					runningThreadCount++;
					modelCount++;
					nextRunningSlot++;
				}
				else if (runningThreadCount > 0) {
					// Joining a local model which was done with learning:
					try {
						learners[waitingThreadPointer].join();
					}
					catch(InterruptedException ie) {
						System.out.println("Join failed: " + ie);
					}
					
					int mp = waitingThreadPointer;
					int mc = completeModelCount;
					completeModelCount++;
					
					// Predicting with the new local model and all previous models:
					SparseMatrix predicted = new SparseMatrix(userCount+1, itemCount+1);
					for (int u = 1; u <= userCount; u++) {
						int[] testItems = testMatrix.getRowRef(u).indexList();
						
						if (testItems != null) {
							for (int i : testItems) {
								double weight = KernelSmoothing.kernelize(getUserSimilarity(anchorUser[mc], u), kernelWidth, kernelType)
												* KernelSmoothing.kernelize(getItemSimilarity(anchorItem[mc], i), kernelWidth, kernelType);
								double newPrediction = learners[mp].getUserFeatures().getRowRef(u).innerProduct(learners[mp].getItemFeatures().getColRef(i)) * weight;
								cumWeight.setValue(u, i, cumWeight.getValue(u, i) + weight);
								cumPrediction.setValue(u, i, cumPrediction.getValue(u, i) + newPrediction);
								double prediction = cumPrediction.getValue(u, i) / cumWeight.getValue(u, i);
								
								if (Double.isNaN(prediction) || prediction == 0.0) {
									prediction = (maxValue + minValue)/2;
								}
								
								if (prediction < minValue) {
									prediction = minValue;
								}
								else if (prediction > maxValue) {
									prediction = maxValue;
								}
								
								predicted.setValue(u, i, prediction);
								
								if (u == anchorUser[mc] && i == anchorItem[mc]) {
									anchorErr = Math.abs(prediction - testMatrix.getValue(u, i));
									anchorErrCum += Math.pow(anchorErr, 2);
								}
							}
						}
					}
					
					EvaluationMetrics e = new EvaluationMetrics(testMatrix, predicted, maxValue, minValue);
					
					if (showProgress) {
						System.out.println((modelCount - multiThreadCount + 1) + "\t" + learners[mp].getAnchorUser() + "\t" + learners[mp].getAnchorItem() + "\t"
							+ learners[mp].getTrainErr() + "\t" + e.getRMSE() + "\t" + anchorErr + "\t" + Math.sqrt(anchorErrCum/(mp+1)));
					}
					
					if (e.getRMSE() < lowestRMSE) {
						lowestRMSE = e.getRMSE();
						preservedEvalMet = e;
					}
					
					nextRunningSlot = waitingThreadPointer;
					waitingThreadPointer = (waitingThreadPointer + 1) % multiThreadCount;
					runningThreadCount--;
					
					// Release memory used for similarity prefetch:
					int au = anchorUser[mc];
					for (int u = 1; u <= userCount; u++) {
						if (au <= u) {
							userSimilarity.setValue(au, u, 0.0);
						}
						else {
							userSimilarity.setValue(u, au, 0.0);
						}
					}
					int ai = anchorItem[mc];
					for (int i = 1; i <= itemCount; i++) {
						if (ai <= i) {
							itemSimilarity.setValue(ai, i, 0.0);
						}
						else {
							itemSimilarity.setValue(i, ai, 0.0);
						}
					}
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
		return preservedEvalMet;
	}
	
	
	
	/**
	 * Calculate similarity between two users, based on the global base SVD.
	 * 
	 * @param idx1 The first user's ID.
	 * @param idx2 The second user's ID.
	 * 
	 * @return The similarity value between two users idx1 and idx2.
	 */
	public double getUserSimilarity (int idx1, int idx2) {
		double sim;
		if (idx1 <= idx2) {
			sim = userSimilarity.getValue(idx1, idx2);
		}
		else {
			sim = userSimilarity.getValue(idx2, idx1);
		}
		
		if (sim == 0.0) {
			SparseVector u_vec = baseline.getU().getRowRef(idx1);
			SparseVector v_vec = baseline.getU().getRowRef(idx2);
			
			sim = 1 - 2.0 / Math.PI * Math.acos(u_vec.innerProduct(v_vec) / (u_vec.norm() * v_vec.norm()));
			
			if (Double.isNaN(sim)) {
				sim = 0.0;
			}
			
			if (idx1 <= idx2) {
				userSimilarity.setValue(idx1, idx2, sim);
			}
			else {
				userSimilarity.setValue(idx2, idx1, sim);
			}
		}
		
		return sim;
	}
	
	/**
	 * Calculate similarity between two items, based on the global base SVD.
	 * 
	 * @param idx1 The first item's ID.
	 * @param idx2 The second item's ID.
	 * 
	 * @return The similarity value between two items idx1 and idx2.
	 */
	public double getItemSimilarity (int idx1, int idx2) {
		double sim;
		if (idx1 <= idx2) {
			sim = itemSimilarity.getValue(idx1, idx2);
		}
		else {
			sim = itemSimilarity.getValue(idx2, idx1);
		} 
		
		if (sim == 0.0) {
			SparseVector i_vec = baseline.getV().getColRef(idx1);
			SparseVector j_vec = baseline.getV().getColRef(idx2);
			
			sim = 1 - 2.0 / Math.PI * Math.acos(i_vec.innerProduct(j_vec) / (i_vec.norm() * j_vec.norm()));
			
			if (Double.isNaN(sim)) {
				sim = 0.0;
			}
			
			if (idx1 <= idx2) {
				itemSimilarity.setValue(idx1, idx2, sim);
			}
			else {
				itemSimilarity.setValue(idx2, idx1, sim);
			}
		}
		
		return sim;
	}
	
	/**
	 * Given the similarity, it applies the given kernel.
	 * This is done either for all users or for all items.
	 * 
	 * @param size The length of user or item vector.
	 * @param id The identifier of anchor point.
	 * @param kernelType The type of kernel.
	 * @param width Kernel width.
	 * @param isItemFeature return item kernel if yes, return user kernel otherwise.
	 * 
	 * @return The kernel-smoothed values for all users or all items.
	 */
	private SparseVector kernelSmoothing(int size, int id, int kernelType, double width, boolean isItemFeature) {
		SparseVector newFeatureVector = new SparseVector(size);
		newFeatureVector.setValue(id, 1.0);
		
		for (int i = 1; i < size; i++) {
			double sim;
			if (isItemFeature) {
				sim = getItemSimilarity(i, id);
			}
			else { // userFeature
				sim = getUserSimilarity(i, id);
			}
			
			newFeatureVector.setValue(i, KernelSmoothing.kernelize(sim, width, kernelType));
		}
		
		return newFeatureVector;
	}
}
