package prea.recommender.matrix;
import prea.data.structure.SparseMatrix;
import prea.util.Distribution;

/**
 * This is a class implementing Probabilistic Matrix Factorization.
 * Technical detail of the algorithm can be found in
 * Ruslan Salakhutdinov and Andriy Mnih, Probabilistic Matrix Factorization,
 * Advances in Neural Information Processing Systems 20, Cambridge, MA: MIT Press, 2008.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class PMF extends MatrixFactorizationRecommender {
	private static final long serialVersionUID = 4003;
	
	/*========================================
	 * Constructors
	 *========================================*/
	/**
	 * Construct a matrix-factorization model with the given data.
	 * 
	 * @param uc The number of users in the dataset.
	 * @param ic The number of items in the dataset.
	 * @param max The maximum rating value in the dataset.
	 * @param min The minimum rating value in the dataset.
	 * @param fc The number of features used for describing user and item profiles.
	 * @param lr Learning rate for gradient-based or iterative optimization.
	 * @param r Controlling factor for the degree of regularization. 
	 * @param m Momentum used in gradient-based or iterative optimization.
	 * @param iter The maximum number of iterations.
	 * @param verbose Indicating whether to show iteration steps and train error.
	 */
	public PMF(int uc, int ic, double max, double min, int fc, double lr, double r, double m, int iter, boolean verbose) {
		super(uc, ic, max, min, fc, lr, r, m, iter, verbose);
	}
	
	/*========================================
	 * Model Builder
	 *========================================*/
	/**
	 * Build a model with given training set.
	 * 
	 * @param rateMatrix Training data set.
	 */
	@Override
	public void buildModel(SparseMatrix rateMatrix) {
		super.buildModel(rateMatrix);
		
		int round = 0;
		double prevErr = 99999;
		double currErr = 9999;
		
		int rateCount = rateMatrix.itemCount();
		double mean_rating = rateMatrix.average();
		this.offset = mean_rating;
		
		SparseMatrix userFeaturesInc = new SparseMatrix(userCount+1, featureCount);
		SparseMatrix itemFeaturesInc = new SparseMatrix(featureCount, itemCount+1);
		
		// Initialize with random values:
		for (int f = 0; f < featureCount; f++) {
			for (int u = 1; u <= userCount; u++) {
				userFeatures.setValue(u, f, 0.1 * Distribution.normalRandom(0, 1));
			}
			for (int i = 1; i <= itemCount; i++) {
				itemFeatures.setValue(f, i, 0.1 * Distribution.normalRandom(0, 1));
			}
		}
		
		// Iteration:
		while (prevErr > currErr && round < maxIter) {
			double errSum = 0.0;
			SparseMatrix userDelta = new SparseMatrix (userCount+1, featureCount);
			SparseMatrix itemDelta = new SparseMatrix (featureCount, itemCount+1);
			
			for (int u = 1; u <= userCount; u++) {
				int[] itemList = rateMatrix.getRowRef(u).indexList();
				if (itemList == null) continue;
				
				for (int i : itemList) {
					// Compute predictions:
					double realRating = rateMatrix.getValue(u, i) - mean_rating;
					double prediction = userFeatures.getRowRef(u).innerProduct(itemFeatures.getColRef(i));
					double userFeatureSum = userFeatures.getRowRef(u).sum();
					double itemFeatureSum = itemFeatures.getColRef(i).sum();
					double err = Math.pow(prediction - realRating, 2) + 0.5 * regularizer * (Math.pow(userFeatureSum, 2) + Math.pow(itemFeatureSum, 2));
					errSum += err;
					
					// Compute gradients:
					double repmatValue = 2 * (prediction - realRating);
					for (int f = 0; f < featureCount; f++) {
						double Ix_p = repmatValue * itemFeatures.getValue(f, i) + regularizer * userFeatures.getValue(u, f);
						double Ix_m = repmatValue * userFeatures.getValue(u, f) + regularizer * itemFeatures.getValue(f, i);
						
						userDelta.setValue(u, f, userDelta.getValue(u, f) + Ix_p);
						itemDelta.setValue(f, i, itemDelta.getValue(f, i) + Ix_m);
					}
				}
			}
			
			// Update user and item features:
			userFeaturesInc = userFeaturesInc.scale(momentum).plus(userDelta.scale(learningRate / rateCount));
		    userFeatures = userFeatures.plus(userFeaturesInc.scale(-1));
		    
		    itemFeaturesInc = itemFeaturesInc.scale(momentum).plus(itemDelta.scale(learningRate / rateCount));
		    itemFeatures = itemFeatures.plus(itemFeaturesInc.scale(-1));
		    
		    
		    round++;
		    
		    // show progress:
		    prevErr = currErr;
		    currErr = errSum/rateCount;
		    
		    if (showProgress)
		    	System.out.println(round + "\t" + currErr);
		}
	}
}
