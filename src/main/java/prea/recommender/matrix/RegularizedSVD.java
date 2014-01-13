package prea.recommender.matrix;
import prea.data.structure.SparseMatrix;
import prea.data.structure.SparseVector;

/**
 * This is a class implementing Regularized SVD (Singular Value Decomposition).
 * Technical detail of the algorithm can be found in
 * Arkadiusz Paterek, Improving Regularized Singular Value Decomposition Collaborative Filtering,
 * Proceedings of KDD Cup and Workshop, 2007.
 * 
 * @author Joonseok Lee
 * @since 2012. 4. 20
 * @version 1.1
 */
public class RegularizedSVD extends MatrixFactorizationRecommender {
	private static final long serialVersionUID = 4001;
	
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
	public RegularizedSVD(int uc, int ic, double max, double min, int fc, double lr, double r, double m, int iter, boolean verbose) {
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
		
		// Gradient Descent:
		int round = 0;
		int rateCount = rateMatrix.itemCount();
		double prevErr = 99999;
		double currErr = 9999;
		
		while (Math.abs(prevErr - currErr) > 0.0001 && round < maxIter) {
			double sum = 0.0;
			for (int u = 1; u <= userCount; u++) {
				SparseVector items = rateMatrix.getRowRef(u);
				int[] itemIndexList = items.indexList();
				
				if (itemIndexList != null) {
					for (int i : itemIndexList) {
						SparseVector Fu = userFeatures.getRowRef(u);
						SparseVector Gi = itemFeatures.getColRef(i);
						
						double AuiEst = Fu.innerProduct(Gi);
						double AuiReal = rateMatrix.getValue(u, i);
						double err = AuiReal - AuiEst;
						sum += Math.abs(err);
						
						for (int s = 0; s < featureCount; s++) {
							double Fus = userFeatures.getValue(u, s);
							double Gis = itemFeatures.getValue(s, i);
							userFeatures.setValue(u, s, Fus + learningRate*(err*Gis - regularizer*Fus));
							itemFeatures.setValue(s, i, Gis + learningRate*(err*Fus - regularizer*Gis));
						}
					}
				}
			}
		
			prevErr = currErr;
			currErr = sum/rateCount;
		
			round++;
			
			// Show progress:
			if (showProgress) {
				System.out.println(round + "\t" + currErr);
			}
		}
	}
}
